"""Event-level behavioural flags and per-(user, hour) window features.

Everything is causal: "new" means never seen earlier in the log, and the
hour-of-day profile is learned from the training period only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SPLIT = PROCESSED / "split.json"
# results/lanl/ and results/synthetic/ live side by side.
RESULTS = ROOT / "results" / (json.loads(SPLIT.read_text())["source"] if SPLIT.exists() else "synthetic")
WINDOW = 3600
# LANL logon types that correspond to Windows 4624 LogonType 2 / 3 / 10.
LOGON_TYPES = {"Interactive", "Network", "RemoteInteractive"}
FEATURES = ["n_events", "n_dst", "n_src", "n_new_pair", "n_new_src", "n_fail",
            "fail_ratio", "n_ntlm", "n_network", "n_rdp", "min_hour_freq"]


def load() -> tuple[pd.DataFrame, dict]:
    """Processed events with behavioural flags, and the period boundaries."""
    events = pd.read_parquet(PROCESSED / "events.parquet")
    split = json.loads(SPLIT.read_text())
    return add_event_flags(events, split), split


def add_event_flags(ev: pd.DataFrame, split: dict) -> pd.DataFrame:
    """Add causal per-event flags: new user->host pair, new source host, hour rarity for this user."""
    ev = ev.sort_values("time", kind="stable").reset_index(drop=True)
    ev["hour_idx"] = ev.time // WINDOW
    ev["hod"] = ev.hour_idx % 24  # LANL's epoch is undisclosed, so this is a relative hour
    ev["fail"] = ev.success.eq("Fail")
    ev["ntlm"] = ev.auth_type.eq("NTLM")
    ev["network"] = ev.logon_type.eq("Network")
    ev["rdp"] = ev.logon_type.eq("RemoteInteractive")
    # "logon" = successful interactive/network/RDP logon, i.e. a 4624 with type 2/3/10.
    ev["logon"] = ev.auth_orientation.eq("LogOn") & ~ev.fail & ev.logon_type.isin(LOGON_TYPES)
    ev["remote"] = ev.logon & ev.src_comp.ne(ev.dst_comp)

    logons = ev[ev.logon]
    ev["new_pair"] = False
    ev["new_src"] = False
    ev.loc[logons.index, "new_pair"] = ~logons.duplicated(["src_user", "dst_comp"]).values
    remote = logons[logons.remote]  # source host matters only when it differs from the target
    ev.loc[remote.index, "new_src"] = ~remote.duplicated(["src_user", "src_comp"]).values

    # Fraction of the user's training-period logons that happen at each hour of day.
    train = logons[logons.time < split["train_end"]]
    counts = train.groupby(["src_user", "hod"]).size()
    freq = (counts / counts.groupby(level=0).transform("sum")).rename("hour_freq")
    ev = ev.merge(freq.reset_index(), on=["src_user", "hod"], how="left")
    ev["hour_freq"] = ev.hour_freq.fillna(0.0)  # unseen user or unseen hour = rarest
    ev.loc[~ev.logon, "hour_freq"] = 1.0  # only logons count towards off-hours behaviour
    return ev


def build_windows(ev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events into one row per (user, hour) with the FEATURES columns and a label."""
    ev = ev.assign(dst_remote=ev.dst_comp.where(ev.remote))
    w = ev.groupby(["src_user", "hour_idx"]).agg(
        n_events=("time", "size"),
        n_dst=("dst_remote", "nunique"),  # distinct hosts reached by remote logons = fan-out
        n_src=("src_comp", "nunique"),
        n_new_pair=("new_pair", "sum"),
        n_new_src=("new_src", "sum"),
        n_fail=("fail", "sum"),
        n_ntlm=("ntlm", "sum"),
        n_network=("network", "sum"),
        n_rdp=("rdp", "sum"),
        min_hour_freq=("hour_freq", "min"),
        label=("label", "max"),
    ).reset_index()
    w["fail_ratio"] = w.n_fail / w.n_events
    w["time"] = w.hour_idx * WINDOW
    return w


def period(df: pd.DataFrame, split: dict) -> pd.Series:
    """Name the period each row belongs to: burn_in / train / calib / test."""
    edges = [split["burn_in_end"], split["train_end"], split["calib_end"]]
    return pd.Series(np.array(["burn_in", "train", "calib", "test"])[np.searchsorted(edges, df.time, side="right")],
                     index=df.index)


if __name__ == "__main__":
    ev, split = load()
    w = build_windows(ev)
    w["period"] = period(w, split)
    print(w.groupby("period").agg(windows=("label", "size"), red=("label", "sum")))
    print(w[FEATURES].describe().T[["mean", "50%", "max"]])
    assert ev.new_pair.sum() <= ev.logon.sum() and w.label.sum() >= 1
