"""Small causal Transformer over each user's sequence of auth events.

Every event is one token, the sum of embeddings of its fields. Reading a user's
events in time order, the model predicts every field of the next event. An
event's anomaly score is its negative log-likelihood ("surprise"), and the
per-field NLL says which part of the event was surprising.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

FIELDS = ["dst_comp", "src_comp", "auth_type", "logon_type", "auth_orientation", "success", "hod"]
SEQ_LEN = 64
MIN_COUNT = 2  # hosts seen fewer times than this in training map to UNK (id 0)


def build_vocab(train_ev: pd.DataFrame) -> dict:
    """Per-field value -> id maps (id 0 = UNK); rare hosts are left out so they map to UNK."""
    vocab = {}
    for f in FIELDS + ["src_user"]:
        counts = train_ev[f].value_counts()
        keep = counts.index[counts >= MIN_COUNT] if f.endswith("_comp") else counts.index
        vocab[f] = {v: i + 1 for i, v in enumerate(keep)}
    return vocab


def make_chunks(ev: pd.DataFrame, vocab: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut each user's time-ordered events (ev must be time-sorted) into chunks of SEQ_LEN.

    Returns codes [C, L, F], users [C], rows [C, L] (positional index into ev, -1 = padding).
    """
    ucode, _ = pd.factorize(ev.src_user)
    pos = ev.groupby("src_user", sort=False).cumcount().to_numpy()
    chunk, _ = pd.factorize(ucode.astype(np.int64) * 10**8 + pos // SEQ_LEN)
    slot = pos % SEQ_LEN
    codes = np.zeros((chunk.max() + 1, SEQ_LEN, len(FIELDS)), np.int64)
    rows = np.full((chunk.max() + 1, SEQ_LEN), -1, np.int64)
    codes[chunk, slot] = np.stack([ev[f].map(vocab[f]).fillna(0).to_numpy(np.int64) for f in FIELDS], 1)
    rows[chunk, slot] = np.arange(len(ev))
    users = np.zeros(chunk.max() + 1, np.int64)
    users[chunk] = ev.src_user.map(vocab["src_user"]).fillna(0).to_numpy(np.int64)
    return codes, users, rows


class Block(nn.Module):
    """Pre-norm Transformer encoder layer that also returns its attention weights."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.n1(x)
        a, w = self.attn(h, h, h, attn_mask=mask, need_weights=True)
        x = x + a
        return x + self.ff(self.n2(x)), w


class EventTransformer(nn.Module):
    def __init__(self, sizes: list[int], n_users: int, d: int = 64, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.field_emb = nn.ModuleList(nn.Embedding(n, d) for n in sizes)
        self.user_emb = nn.Embedding(n_users, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList(nn.Linear(d, n) for n in sizes)
        self.register_buffer("mask", torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), 1))

    def forward(self, codes: torch.Tensor, users: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """codes [B, L, F] -> per-field NLL [B, L, F] of each event given the earlier ones,
        and last-layer attention [B, L, L]. Input position 0 is a start token carrying only
        the user embedding; input position i > 0 is event i-1."""
        tok = sum(emb(codes[..., i]) for i, emb in enumerate(self.field_emb))
        u = self.user_emb(users)[:, None]
        x = torch.cat([u, tok[:, :-1] + u], 1) + self.pos_emb.weight
        for block in self.blocks:
            x, attn = block(x, self.mask)
        x = self.norm(x)
        nll = torch.stack([F.cross_entropy(head(x).transpose(1, 2), codes[..., i], reduction="none")
                           for i, head in enumerate(self.heads)], -1)
        return nll, attn


def train(train_ev: pd.DataFrame, val_ev: pd.DataFrame, epochs: int = 20, patience: int = 2,
          batch: int = 64, lr: float = 2e-3, seed: int = 0) -> dict:
    """Fit on train_ev; keep the epoch with the lowest NLL on the later, unlabeled val_ev."""
    torch.manual_seed(seed)
    vocab = build_vocab(train_ev)
    codes, users, rows = make_chunks(train_ev, vocab)
    sizes = [len(vocab[f]) + 1 for f in FIELDS]
    model = EventTransformer(sizes, len(vocab["src_user"]) + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    codes, users, valid = torch.from_numpy(codes), torch.from_numpy(users), torch.from_numpy(rows >= 0)
    print(f"  {len(codes):,} chunks of {SEQ_LEN} events, {sum(p.numel() for p in model.parameters()):,} parameters")
    best, best_state, since_best = np.inf, None, 0
    for epoch in range(epochs):
        perm, total = torch.randperm(len(codes)), 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            nll, _ = model(codes[idx], users[idx])
            loss = nll.sum(-1)[valid[idx]].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        bundle = dict(model=model.eval(), vocab=vocab, sizes=sizes)
        val = score_events(bundle, val_ev).sum(1).mean()
        model.train()
        print(f"  epoch {epoch + 1}/{epochs}: NLL per event train {total / len(perm):.3f}, val {val:.3f}")
        if val < best:
            best, best_state, since_best = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    print(f"  kept the epoch with val NLL {best:.3f}")
    return dict(model=model.eval(), vocab=vocab, sizes=sizes)


@torch.no_grad()
def score_events(bundle: dict, ev: pd.DataFrame, batch: int = 256) -> np.ndarray:
    """Per-field NLL for every row of the time-sorted ev, shape [N, len(FIELDS)]."""
    codes, users, rows = make_chunks(ev, bundle["vocab"])
    out = np.zeros((len(ev), len(FIELDS)), np.float32)
    for i in range(0, len(codes), batch):
        nll, _ = bundle["model"](torch.from_numpy(codes[i:i + batch]), torch.from_numpy(users[i:i + batch]))
        r = rows[i:i + batch]
        out[r[r >= 0]] = nll.numpy()[r >= 0]
    return out


def save(bundle: dict, path: Path) -> None:
    torch.save({"state": bundle["model"].state_dict(), "vocab": bundle["vocab"], "sizes": bundle["sizes"]}, path)


def load(path: Path) -> dict:
    ckpt = torch.load(path, weights_only=False)  # our own file; vocab holds plain Python dicts
    model = EventTransformer(ckpt["sizes"], len(ckpt["vocab"]["src_user"]) + 1)
    model.load_state_dict(ckpt["state"])
    return dict(model=model.eval(), vocab=ckpt["vocab"], sizes=ckpt["sizes"])
