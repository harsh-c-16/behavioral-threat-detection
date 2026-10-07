"""Turn an alert into its top reasons.

Isolation Forest: SHAP (TreeExplainer) on the window features.
Transformer: per-field surprise (NLL) of the most surprising event in the
window, plus the earlier events the last attention layer looked at most.
"""
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import IsolationForest

import model_iforest
from features import FEATURES
from model_transformer import FIELDS, make_chunks


def iforest_reasons(model: IsolationForest, windows: pd.DataFrame, top: int = 3) -> list[list[dict]]:
    """For each window, the features pushing it most towards 'anomalous'.

    SHAP explains the forest's expected path length; a negative value means the
    feature made the window easier to isolate, i.e. more anomalous.
    """
    values = shap.TreeExplainer(model).shap_values(model_iforest.matrix(windows))
    out = []
    for row, sv in zip(windows[FEATURES].itertuples(index=False), values):
        order = np.argsort(sv)[:top]
        out.append([{"feature": FEATURES[i], "value": round(float(row[i]), 4), "shap": round(float(sv[i]), 4)}
                    for i in order if sv[i] < 0])
    return out


@torch.no_grad()
def transformer_reasons(bundle: dict, ev: pd.DataFrame, nll: np.ndarray, rows: np.ndarray,
                        top: int = 3) -> list[dict]:
    """Explain the events at positional `rows` of ev, given per-field nll [N, F]."""
    codes, users, chunk_rows = make_chunks(ev, bundle["vocab"])
    chunk_of, slot_of = np.zeros(len(ev), np.int64), np.zeros(len(ev), np.int64)
    c_idx, s_idx = np.nonzero(chunk_rows >= 0)
    chunk_of[chunk_rows[c_idx, s_idx]], slot_of[chunk_rows[c_idx, s_idx]] = c_idx, s_idx
    out = []
    for r in rows:
        c, s = chunk_of[r], slot_of[r]
        _, attn = bundle["model"](torch.from_numpy(codes[c:c + 1]), torch.from_numpy(users[c:c + 1]))
        weights = attn[0, s, :s + 1].numpy()  # input position 0 = user start token, j > 0 = event j-1
        context = []
        for j in np.argsort(weights)[::-1][:top]:
            if j == 0:
                context.append({"attended": "user baseline (start token)", "weight": round(float(weights[j]), 3)})
            else:
                e = ev.iloc[chunk_rows[c, j - 1]]
                context.append({"attended": f"t={e.time} {e.src_comp}->{e.dst_comp} {e.auth_type}/{e.logon_type}",
                                "weight": round(float(weights[j]), 3)})
        event = ev.iloc[r]
        fields = [{"field": FIELDS[i], "value": str(event[FIELDS[i]]), "surprise_nats": round(float(nll[r, i]), 2),
                   "p": float(f"{np.exp(-nll[r, i]):.3g}")} for i in np.argsort(nll[r])[::-1][:top]]
        out.append({"event": f"t={event.time} {event.src_comp}->{event.dst_comp} {event.auth_type}/"
                             f"{event.logon_type}/{event.auth_orientation}/{event.success}",
                    "fields": fields, "attention": context})
    return out
