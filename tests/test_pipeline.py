"""Small, fast checks of data loading, causal features, rules, thresholds and metrics."""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import baseline_rules
import evaluate
import features
import prepare_data
from download_data import LogClock

DAY = 86400


@pytest.fixture(scope="module")
def synthetic():
    auth, red = prepare_data.generate_synthetic(seed=1, n_users=40, n_days=14, n_campaigns=6)
    return prepare_data.label_events(auth, red), red


def test_synthetic_labels_join_and_stay_in_test_days(synthetic):
    events, red = synthetic
    assert list(events.columns) == prepare_data.AUTH_COLS + ["label"]
    assert events.label.sum() == len(red.drop_duplicates())  # every red-team hop matches an auth row
    assert events.time[events.label == 1].min() >= 11 * DAY  # campaigns only in the last 3 days


def test_truncated_gzip_is_read_up_to_the_cut(tmp_path):
    rng = np.random.default_rng(0)
    lines = [f"{t},U{rng.integers(1000)}@D,U1@D,C{rng.integers(10**5)},C{rng.integers(10**5)},NTLM,Network,LogOn,Success\n"
             for t in range(50_000)]
    blob = gzip.compress("".join(lines).encode())
    path = tmp_path / "auth.txt.gz"
    path.write_bytes(blob[: len(blob) // 2])
    auth = prepare_data.read_auth(path, max_time=10**9, keep_permille=1000, always=set())
    assert 0 < len(auth) < 50_000
    assert (auth.time.diff().dropna() == 1).all()  # no partial or reordered rows


def test_log_clock_tracks_last_complete_line():
    blob = gzip.compress(b"5,a\n6,b\n7,c\n8,partial")
    clock = LogClock()
    for i in range(0, len(blob), 7):
        clock.feed(blob[i:i + 7])
    assert clock.last_time == 7


def test_new_pair_is_causal():
    ev = pd.DataFrame({"time": [10, 20, 30], "src_user": "U1", "dst_user": "U1", "src_comp": "C1",
                       "dst_comp": ["C2", "C2", "C3"], "auth_type": "Kerberos", "logon_type": "Network",
                       "auth_orientation": "LogOn", "success": "Success", "label": 0})
    split = dict(burn_in_end=0, train_end=100, calib_end=200, test_end=300)
    flags = features.add_event_flags(ev, split)
    assert flags.new_pair.tolist() == [True, False, True]
    # Appending a later event must not change any earlier flag.
    later = pd.concat([ev, ev.iloc[[0]].assign(time=40)], ignore_index=True)
    assert features.add_event_flags(later, split).new_pair.tolist()[:3] == [True, False, True]


def test_periods_and_windows(synthetic):
    events, _ = synthetic
    split = dict(burn_in_end=2 * DAY, train_end=9 * DAY, calib_end=11 * DAY, test_end=14 * DAY)
    w = features.build_windows(features.add_event_flags(events, split))
    p = features.period(w, split)
    assert set(p) == {"burn_in", "train", "calib", "test"}
    assert w.label[p != "test"].sum() == 0
    assert (w[features.FEATURES] >= 0).all().all()


def test_fail_then_success_needs_failures_before_success():
    toy = pd.DataFrame({"src_user": "U1", "dst_comp": "C9", "hour_idx": 0, "auth_orientation": "LogOn",
                        "time": [1, 2, 3, 4, 5], "fail": [True] * 4 + [False]})
    assert baseline_rules.fail_then_success(toy).iloc[0] == 1
    success_first = toy.assign(fail=[False] + [True] * 4)
    assert baseline_rules.fail_then_success(success_first).empty


def test_calibrated_threshold_hits_the_alert_budget():
    scores = pd.Series(np.random.default_rng(0).normal(size=100_000))
    thr = evaluate.calibrate(scores, rate=0.005)
    assert (scores > thr).mean() == pytest.approx(0.005, abs=1e-4)


def test_operating_point_counts():
    y = np.array([1, 1, 0, 0, 0, 0], bool)
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    op = evaluate.operating_point(y, s, thr=0.5, days=2)
    assert (op["tp"], op["fp"], op["fn"]) == (1, 1, 1)
    assert op["precision"] == 0.5 and op["recall"] == 0.5 and op["fpr"] == 0.25
    assert op["alerts_per_day"] == 1.0


def test_fp_at_recall():
    y = np.array([1, 0, 1, 0, 0], bool)
    s = np.array([5, 4, 3, 2, 1])
    assert evaluate.fp_at_recall(y, s, 0.5) == 0
    assert evaluate.fp_at_recall(y, s, 1.0) == 1


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    s = y + rng.normal(size=5000)
    ci = evaluate.bootstrap_ci(y, s, thr=1.0, n=200)
    point = evaluate.operating_point(y, s, 1.0, days=1)
    for k in ("pr_auc", "precision", "recall"):
        assert ci[k][0] <= point[k] <= ci[k][1]
