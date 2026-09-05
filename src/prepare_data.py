"""Build the labeled event table used by every model.

Both paths produce LANL-format files first, then go through the same parser:
  --synthetic : generate a LANL-shaped auth log with injected red-team campaigns
  (default)   : read the real LANL auth.txt.gz slice fetched by download_data.py

Output: data/processed/events.parquet and data/processed/split.json
"""
import argparse
import gzip
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DAY = 86400
AUTH_COLS = ["time", "src_user", "dst_user", "src_comp", "dst_comp",
             "auth_type", "logon_type", "auth_orientation", "success"]
RT_COLS = ["time", "src_user", "src_comp", "dst_comp"]


def generate_synthetic(seed: int = 7, n_users: int = 400, n_days: int = 14,
                       n_campaigns: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (auth_df, redteam_df) in LANL format. Time 0 is midnight of day 0.

    Benign noise is deliberate so that the rules are not trivially perfect:
    admins fan out to many hosts, a few users work nights, people sometimes
    visit a server for the first time, legacy servers only speak NTLM, and
    typos produce failed-then-successful logons.
    """
    rng = np.random.default_rng(seed)
    servers = [f"C{i}" for i in range(1000, 1120)]
    popularity = 1.0 / np.arange(1, len(servers) + 1)
    popularity /= popularity.sum()
    dcs = ["C1", "C2"]
    ntlm_servers = set(rng.choice(servers, 12, replace=False))
    rows, red = [], []

    def ev(t, user, src, dst, auth, logon, orient, ok=True):
        rows.append((int(t), user, user, src, dst, auth, logon, orient,
                     "Success" if ok else "Fail"))

    def logon(t, user, src, dst, ok=True, auth=None, logon_type="Network"):
        auth = auth or ("NTLM" if dst in ntlm_servers else "Kerberos")
        if auth == "Kerberos":
            ev(t, user, src, rng.choice(dcs), "Kerberos", "?", "TGS")
        ev(t, user, src, dst, auth, logon_type, "LogOn", ok)
        if ok:
            ev(t + rng.integers(60, 3600), user, src, dst, auth, logon_type, "LogOff")

    users = []
    for u in range(n_users):
        admin = u < 20
        start = rng.normal(20, 1) if rng.random() < 0.05 else rng.normal(8.5, 1)
        users.append(dict(
            name=f"U{u + 10}@DOM1", ws=f"C{u + 2000}", admin=admin,
            usual=list(rng.choice(servers, rng.integers(10, 30) if admin else rng.integers(2, 6),
                                  replace=False, p=popularity)),
            start=start, length=rng.normal(9, 1)))

    for day in range(n_days):
        weekend = day % 7 in (5, 6)
        for u in users:
            if rng.random() > (0.2 if weekend else 0.95):
                continue
            for _ in range(rng.poisson(6) + 1):
                hour = rng.uniform(0, 24) if rng.random() < 0.03 else u["start"] + rng.uniform(0, u["length"])
                t = day * DAY + hour * 3600
                # hot-desking: 2% of sessions start from someone else's workstation
                ws = f"C{rng.integers(2000, 2000 + n_users)}" if rng.random() < 0.02 else u["ws"]
                ev(t, u["name"], ws, ws, "Negotiate", "Interactive", "LogOn")
                ev(t + 1, u["name"], ws, rng.choice(dcs), "Kerberos", "?", "TGT")
                maintenance = u["admin"] and rng.random() < 0.3
                n_targets = rng.integers(8, 20) if maintenance else rng.integers(1, 4)
                for _ in range(n_targets):
                    t += rng.integers(5, 180 if maintenance else 1200)
                    dst = rng.choice(servers) if rng.random() < 0.03 else rng.choice(u["usual"])
                    if rng.random() < 0.02:  # typo'd password, then success
                        for _ in range(rng.integers(1, 5)):
                            logon(t, u["name"], ws, dst, ok=False)
                            t += rng.integers(2, 20)
                    rdp = u["admin"] and rng.random() < 0.1
                    logon(t, u["name"], ws, dst, logon_type="RemoteInteractive" if rdp else "Network")

    # Red team: campaigns only in the last 3 days (the test period).
    workstations = [u["ws"] for u in users]
    for c in range(n_campaigns):
        victim = users[rng.integers(0, n_users)] if c % 4 else users[rng.integers(0, 20)]
        style = ["loud", "stealthy", "rdp"][c % 3]
        day = rng.integers(n_days - 3, n_days)
        hour = rng.uniform(1, 5) if style == "loud" else victim["start"] + rng.uniform(1, 6)
        t = day * DAY + hour * 3600
        current = victim["ws"] if rng.random() < 0.7 else "C3000"
        n_hops = {"loud": rng.integers(8, 15), "stealthy": rng.integers(3, 6), "rdp": rng.integers(3, 7)}[style]
        for _ in range(n_hops):
            t += rng.integers(30, 400) if style == "loud" else rng.integers(600, 5400)
            pool = servers + workstations
            target = pool[rng.integers(0, len(pool))]
            if target in victim["usual"] or target == current:
                continue
            auth = "NTLM" if style == "loud" or rng.random() < 0.3 else "Kerberos"
            for _ in range(rng.integers(0, 4) if style == "loud" else 0):
                logon(t, victim["name"], current, target, ok=False, auth=auth)
                t += rng.integers(1, 10)
            logon(t, victim["name"], current, target, auth=auth,
                  logon_type="RemoteInteractive" if style == "rdp" else "Network")
            red.append((int(t), victim["name"], current, target))
            if rng.random() < 0.5:
                current = target  # pivot: next hop starts from the newly owned host

    auth_df = pd.DataFrame(rows, columns=AUTH_COLS).sort_values("time", kind="stable")
    return auth_df, pd.DataFrame(red, columns=RT_COLS)


def keep_user(users: pd.Series, keep_permille: int, always: set) -> pd.Series:
    """Deterministic user sample: crc32 hash bucket, plus every red-team user."""
    sampled = users.map(lambda u: zlib.crc32(u.encode()) % 1000 < keep_permille).astype(bool)
    return sampled | users.isin(always)


class TruncatedGzip(gzip.GzipFile):
    """GzipFile that treats a cut-off stream (partial download) as a normal end of file."""

    def read(self, size: int = -1) -> bytes:
        try:
            return super().read(size)
        except EOFError:
            return b""

    def read1(self, size: int = -1) -> bytes:
        try:
            return super().read1(size)
        except EOFError:
            return b""


def read_auth(path: Path, max_time: int, keep_permille: int, always: set) -> pd.DataFrame:
    """Stream a (possibly truncated) LANL auth.txt.gz, keep U* user accounts only."""
    parts = []
    with TruncatedGzip(path) as f:
        for chunk in pd.read_csv(f, names=AUTH_COLS, dtype=str, chunksize=2_000_000):
            chunk = chunk.dropna()  # a truncated download can leave a partial last line
            chunk = chunk[chunk.time.str.isdigit()]
            chunk["time"] = chunk.time.astype(np.int64)
            t_last = chunk.time.max()
            chunk = chunk[(chunk.time < max_time) & chunk.src_user.str.match(r"U\d+@")]
            users = pd.Series(chunk.src_user.unique())
            kept = set(users[keep_user(users, keep_permille, always).values])
            parts.append(chunk[chunk.src_user.isin(kept)])
            print(f"  read up to t={t_last:,}, kept {sum(map(len, parts)):,} rows")
            if t_last >= max_time:  # LANL is time-sorted, so nothing useful follows
                break
    return pd.concat(parts, ignore_index=True)


def label_events(auth: pd.DataFrame, redteam: pd.DataFrame) -> pd.DataFrame:
    """label=1 for auth rows matching a red-team (time, user, src, dst) tuple."""
    key = redteam[RT_COLS].drop_duplicates().assign(label=1)
    out = auth.merge(key, on=RT_COLS, how="left")
    out["label"] = out.label.fillna(0).astype(np.int8)
    matched = key.merge(auth[RT_COLS].drop_duplicates(), on=RT_COLS).shape[0]
    print(f"red-team tuples in window: {len(key)}, matched to auth rows: {matched}, "
          f"labeled auth rows: {int(out.label.sum())}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic LANL-shaped data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep-permille", type=int, default=100,
                    help="real data: keep this many users per 1000 (red-team users always kept)")
    # Days 3-4 of LANL are a weekend (about half the users), so train and calibration use weekdays 1 and 2.
    ap.add_argument("--burn-in", type=float, default=1.0, help="history only, not used to train IF")
    ap.add_argument("--train-end", type=float, default=2.0)
    ap.add_argument("--calib-end", type=float, default=3.0, help="benign period for thresholds")
    ap.add_argument("--test-end", type=float, default=10.0)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        auth_path, rt_path = RAW / "synthetic_auth.txt.gz", RAW / "synthetic_redteam.txt.gz"
        auth_df, rt_df = generate_synthetic(seed=args.seed)
        auth_df.to_csv(auth_path, header=False, index=False)
        rt_df.to_csv(rt_path, header=False, index=False)
        split = dict(source="synthetic", burn_in_end=2 * DAY, train_end=9 * DAY,
                     calib_end=11 * DAY, test_end=14 * DAY)
        keep_permille = 1000
    else:
        auth_path, rt_path = RAW / "auth.txt.gz", RAW / "redteam.txt.gz"
        if not auth_path.exists() or not rt_path.exists():
            raise SystemExit(f"missing {auth_path} or {rt_path}: run src/download_data.py first, "
                             "or use --synthetic")
        split = dict(source="lanl", burn_in_end=int(args.burn_in * DAY), train_end=int(args.train_end * DAY),
                     calib_end=int(args.calib_end * DAY), test_end=int(args.test_end * DAY))
        keep_permille = args.keep_permille
    assert split["burn_in_end"] < split["train_end"] < split["calib_end"] < split["test_end"], split

    redteam = pd.read_csv(rt_path, names=RT_COLS)
    redteam = redteam[redteam.time < split["test_end"]]
    print(f"reading {auth_path.name} up to day {split['test_end'] / DAY:g}")
    auth = read_auth(auth_path, split["test_end"], keep_permille, set(redteam.src_user))
    events = label_events(auth, redteam).sort_values("time", kind="stable").reset_index(drop=True)

    in_calib = events.label.astype(bool) & (events.time >= split["train_end"]) & (events.time < split["calib_end"])
    in_test = events.label.astype(bool) & (events.time >= split["calib_end"])
    if in_calib.any():
        print(f"WARNING: {int(in_calib.sum())} red-team rows fall in the calibration period; "
              "their windows are excluded from threshold calibration")
    split.update(n_events=len(events), n_users=int(events.src_user.nunique()),
                 n_computers=int(pd.concat([events.src_comp, events.dst_comp]).nunique()),
                 n_red_events_test=int(in_test.sum()))
    events.to_parquet(PROCESSED / "events.parquet", index=False)
    (PROCESSED / "split.json").write_text(json.dumps(split, indent=2))
    print(json.dumps(split, indent=2))


if __name__ == "__main__":
    main()
