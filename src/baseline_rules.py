"""Rule/statistical baseline: pure pandas, no ML.

Each rule is the pandas equivalent of one KQL file in detections/. A window's
score is the number of distinct rules that fired in it (ties broken by how many
events were flagged), so the baseline can be ranked and thresholded like the
ML models.
"""
import pandas as pd

FANOUT_K = 10            # distinct remote hosts per account per hour
OFF_HOURS_FREQ = 0.01    # user's baseline share of logons at this hour of day
FAIL_K = 3               # failures before a success on the same host

# rule name -> detections/*.kql file implementing the same logic
RULES = {
    "fanout": "01_fanout_many_hosts.kql",
    "first_pair": "02_first_user_host_pair.kql",
    "off_hours": "03_off_hours_logon.kql",
    "fail_then_success": "04_failed_then_success.kql",
    "ntlm_new_host": "05_ntlm_network_new_host.kql",
    "rdp_new_host": "06_rdp_new_host.kql",
    "new_src": "07_new_source_host.kql",
}


def event_rules(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-event boolean flags for the rules that look at single logons."""
    return pd.DataFrame({
        "first_pair": ev.new_pair & ev.remote,
        "off_hours": ev.logon & (ev.hour_freq < OFF_HOURS_FREQ),
        "ntlm_new_host": ev.new_pair & ev.ntlm & ev.network,
        "rdp_new_host": ev.new_pair & ev.rdp,
        "new_src": ev.new_src,
    })


def fail_then_success(ev: pd.DataFrame) -> pd.Series:
    """(user, hour) windows where one host saw >= FAIL_K failures followed by a success."""
    tries = ev[ev.auth_orientation.eq("LogOn")]
    g = tries.assign(fail_time=tries.time.where(tries.fail),
                     ok_time=tries.time.where(~tries.fail)).groupby(["src_user", "dst_comp", "hour_idx"])
    agg = g.agg(n_fail=("fail", "sum"), first_fail=("fail_time", "min"), last_ok=("ok_time", "max"))
    hit = agg[(agg.n_fail >= FAIL_K) & (agg.last_ok > agg.first_fail)].reset_index()
    return hit.groupby(["src_user", "hour_idx"]).size().rename("fail_then_success")


def score_windows(ev: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Return windows with one 0/1 column per rule plus `score`."""
    flags = event_rules(ev)
    per_window = flags.groupby([ev.src_user, ev.hour_idx]).sum()
    out = windows.join(per_window, on=["src_user", "hour_idx"])
    out = out.join(fail_then_success(ev), on=["src_user", "hour_idx"]).fillna({"fail_then_success": 0})
    out["fanout"] = (out.n_dst >= FANOUT_K).astype(int)
    fired = (out[list(RULES)] > 0)
    n_flagged = out[list(RULES)].sum(axis=1)
    out["score"] = fired.sum(axis=1) + n_flagged / (n_flagged + 1)
    out["rules_fired"] = fired.apply(lambda r: [k for k in RULES if r[k]], axis=1)
    return out


if __name__ == "__main__":
    from features import build_windows, load, period
    ev, split = load()
    w = score_windows(ev, build_windows(ev))
    w = w[period(w, split) == "test"]
    print(pd.DataFrame({r: [int((w[r] > 0).sum()), int(((w[r] > 0) & (w.label == 1)).sum())] for r in RULES},
                       index=["windows_fired", "of_which_red"]).T)
