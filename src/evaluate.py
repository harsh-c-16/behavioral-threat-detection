"""Score all three models, set thresholds on the benign calibration period, report on test.

Writes results/<source>/{comparison.csv, comparison.md, ci95.csv, fp_reduction.csv,
metrics.json, pr_curve.png, alerts.jsonl}.
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import baseline_rules
import explain
import model_iforest
import model_transformer
from attack_mapping import techniques_for
from features import RESULTS, ROOT, build_windows, load, period

ALERT_RATE = 0.005  # thresholds alert on 0.5% of benign calibration windows
TOP_K = 3  # transformer window score sums this many of the most surprising events
N_BOOT = 1000
MODELS = {"rules": "Rule baseline", "iforest": "Isolation Forest", "transformer": "Transformer"}
COLORS = {"rules": "#2a78d6", "iforest": "#eb6834", "transformer": "#1baf7a"}


def calibrate(scores: pd.Series, rate: float = ALERT_RATE) -> float:
    """Threshold that alerts on `rate` of the (benign, calibration-period) scores."""
    return float(np.quantile(scores, 1 - rate))


def operating_point(y: np.ndarray, s: np.ndarray, thr: float, days: float) -> dict:
    """Confusion counts and rates when alerting on s > thr, plus threshold-free PR-AUC / ROC-AUC."""
    alert = s > thr
    tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
    fn, tn = int((~alert & y).sum()), int((~alert & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(threshold=float(thr), alerts=tp + fp, tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=2 * p * r / (p + r) if p + r else 0.0, fpr=fp / (fp + tn),
                alerts_per_day=(tp + fp) / days, fp_per_day=fp / days,
                pr_auc=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)))


def fp_at_recall(y: np.ndarray, s: np.ndarray, target: float) -> int:
    """Fewest false positives a model needs to reach `target` recall (ties count in full)."""
    p, r, _ = precision_recall_curve(y, s)
    ok = (r >= target) & (p > 0)
    tps = r[ok] * y.sum()
    return int(np.round(np.min(tps * (1 - p[ok]) / p[ok])))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, thr: float, n: int = N_BOOT, seed: int = 0) -> dict:
    """95% percentile intervals of PR-AUC, precision and recall, resampling test windows with replacement."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yi, alert = y[i], s[i] > thr
        if not yi.any():
            continue
        tp = (alert & yi).sum()
        stats.append((average_precision_score(yi, s[i]), tp / max(alert.sum(), 1), tp / yi.sum()))
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {k: [float(a), float(b)] for k, a, b in zip(("pr_auc", "precision", "recall"), lo, hi)}


def score_all(ev: pd.DataFrame, w: pd.DataFrame, iforest: IsolationForest, bundle: dict):
    """Add score_rules / score_iforest / score_transformer to the windows; also returns per-field NLL."""
    w = baseline_rules.score_windows(ev, w).rename(columns={"score": "score_rules"})
    w["score_iforest"] = model_iforest.score(iforest, w)
    nll = model_transformer.score_events(bundle, ev)
    ev["surprise"] = nll.sum(1)
    # Window score = sum of its 3 most surprising events: lateral movement is several
    # unusual hops in a row, while benign novelty is usually a single odd event.
    by_surprise = ev.sort_values("surprise", ascending=False).groupby(["src_user", "hour_idx"])
    agg = pd.DataFrame({"score_transformer": by_surprise.head(TOP_K).groupby(["src_user", "hour_idx"]).surprise.sum(),
                        "top_event": by_surprise.surprise.idxmax()})
    return w.join(agg, on=["src_user", "hour_idx"]), nll


def score_and_threshold(ev: pd.DataFrame, split: dict, iforest: IsolationForest, bundle: dict):
    """Score every window, set each model's threshold on benign calibration windows.

    Returns (test windows, thresholds, per-field NLL of every event).
    """
    w = build_windows(ev)
    w["period"] = period(w, split)
    w, nll = score_all(ev, w, iforest, bundle)
    calib = w[(w.period == "calib") & (w.label == 0)]  # drop any red-team window from calibration
    thresholds = {m: calibrate(calib[f"score_{m}"]) for m in MODELS}
    return w[w.period == "test"], thresholds, nll


def write_alerts(ev: pd.DataFrame, test: pd.DataFrame, thresholds: dict, iforest: IsolationForest,
                 bundle: dict, nll: np.ndarray) -> int:
    """Write one explained alert per (model, window) above threshold to alerts.jsonl; return the count."""
    records = []
    for m in MODELS:
        alerts = test[test[f"score_{m}"] > thresholds[m]]
        if m == "rules":
            reasons = [[{"rule": r, "kql": baseline_rules.RULES[r]} for r in fired] for fired in alerts.rules_fired]
            signals = [[x["rule"] for x in rs] for rs in reasons]
        elif m == "iforest":
            reasons = explain.iforest_reasons(iforest, alerts)
            signals = [[x["feature"] for x in rs] for rs in reasons]
        else:
            reasons = explain.transformer_reasons(bundle, ev, nll, alerts.top_event.to_numpy())
            signals = [[x["field"] for x in rs["fields"]] for rs in reasons]
        for (_, a), rs, sig in zip(alerts.iterrows(), reasons, signals):
            records.append(dict(model=m, user=a.src_user, window_start=int(a.time), score=round(float(a[f"score_{m}"]), 4),
                                threshold=round(float(thresholds[m]), 4), red_team=bool(a.label),
                                reasons=rs, attack=techniques_for(sig)))
    with open(RESULTS / "alerts.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    return len(records)


def plot_pr(test: pd.DataFrame, ops: dict) -> None:
    y = test.label.to_numpy(bool)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    for m, name in MODELS.items():
        p, r, _ = precision_recall_curve(y, test[f"score_{m}"])
        ax.step(r, p, where="post", color=COLORS[m], lw=2, label=f"{name} (AP {ops[m]['pr_auc']:.3f})")
        ax.plot(ops[m]["recall"], ops[m]["precision"], "o", ms=8, color=COLORS[m], mec="white", mew=2)
    ax.axhline(y.mean(), color="#8a8984", lw=1, ls="--", label=f"Base rate ({y.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02),
           title="Red-team (user, hour) windows, test period\ndots = operating point at calibrated threshold")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS / "pr_curve.png")


def main() -> None:
    ev, split = load()
    with open(RESULTS / "iforest.pkl", "rb") as f:
        iforest = pickle.load(f)
    bundle = model_transformer.load(RESULTS / "transformer.pt")
    test, thresholds, nll = score_and_threshold(ev, split, iforest, bundle)
    y = test.label.to_numpy(bool)
    days = (split["test_end"] - split["calib_end"]) / 86400
    ops = {m: operating_point(y, test[f"score_{m}"].to_numpy(), thresholds[m], days) for m in MODELS}
    ci = {m: bootstrap_ci(y, test[f"score_{m}"].to_numpy(), thresholds[m]) for m in MODELS}

    # Compare FPs at each baseline's own recall; matching the Transformer's recall would inflate the gap.
    fp_rows = []
    for m in ("rules", "iforest"):
        ref = ops[m]["recall"]
        fp_t = fp_at_recall(y, test.score_transformer.to_numpy(), ref)
        fp_m = fp_at_recall(y, test[f"score_{m}"].to_numpy(), ref)
        fp_rows.append(dict(model=m, recall=ref, fp_transformer=fp_t, fp_model=fp_m,
                            reduction_pct=100 * (1 - fp_t / fp_m) if fp_m else 0.0))
    pd.DataFrame(fp_rows).to_csv(RESULTS / "fp_reduction.csv", index=False)

    table = pd.DataFrame(ops).T.rename(index=MODELS)
    cols = ["precision", "recall", "f1", "fpr", "alerts_per_day", "fp_per_day", "pr_auc", "roc_auc", "tp", "fp", "alerts"]
    table[cols].to_csv(RESULTS / "comparison.csv")
    fmt = table[cols].astype(float).round(4)
    for c in ("tp", "fp", "alerts"):
        fmt[c] = fmt[c].astype(int)
    md = ["| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    md += [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in zip(fmt.index, fmt.astype(object).values)]
    ci_rows = [{"model": MODELS[m], **{f"{k}_{b}": round(v[j], 4) for k, v in ci[m].items() for j, b in enumerate(("lo", "hi"))}}
               for m in MODELS]
    pd.DataFrame(ci_rows).to_csv(RESULTS / "ci95.csv", index=False)
    md += ["", f"95% bootstrap intervals ({N_BOOT} resamples of test windows):", "",
           "| Model | PR-AUC | precision | recall |", "|---|---|---|---|"]
    md += [f"| {MODELS[m]} | " + " | ".join(f"{ci[m][k][0]:.3f}–{ci[m][k][1]:.3f}" for k in ("pr_auc", "precision", "recall")) + " |"
           for m in MODELS]
    (RESULTS / "comparison.md").write_text("\n".join(md) + "\n")

    # Event-level ranking and window sizes, to see where the window aggregation loses signal.
    te = (ev.time >= split["calib_end"]).to_numpy()
    y_ev = ev.label.to_numpy(bool)[te]
    diagnostics = dict(test_events=int(te.sum()), test_red_events=int(y_ev.sum()), window_base_rate=float(y.mean()),
                       transformer_event_pr_auc=float(average_precision_score(y_ev, ev.surprise[te])),
                       transformer_event_roc_auc=float(roc_auc_score(y_ev, ev.surprise[te])),
                       median_events_red_window=float(test.n_events[y].median()),
                       median_events_benign_window=float(test.n_events[~y].median()))

    n_alerts = write_alerts(ev, test, thresholds, iforest, bundle, nll)
    plot_pr(test, ops)
    kql = sorted((ROOT / "detections").glob("*.kql"))
    metrics = dict(dataset={k: split[k] for k in ("source", "n_events", "n_users", "n_computers")},
                   periods_days={k: split[k] / 86400 for k in ("burn_in_end", "train_end", "calib_end", "test_end")},
                   alert_rate=ALERT_RATE, test_days=days, test_windows=len(test), test_red_windows=int(y.sum()),
                   models=ops, ci95=ci, fp_reduction=fp_rows, diagnostics=diagnostics, alerts_written=n_alerts,
                   n_kql_rules=len(kql), kql_techniques=[t.split()[0] for t in techniques_for(baseline_rules.RULES)])
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nData: {split['source']}, {split['n_events']:,} events; test: {len(test):,} windows, "
          f"{int(y.sum())} red-team, {days:g} days")
    print((RESULTS / "comparison.md").read_text())
    for r in fp_rows:
        print(f"Transformer vs {MODELS[r['model']]} at recall {r['recall']:.3f}: "
              f"{r['fp_transformer']} vs {r['fp_model']} FPs ({r['reduction_pct']:.1f}% reduction)")
    print("diagnostics:", json.dumps({k: round(v, 4) for k, v in diagnostics.items()}))
    print(f"{n_alerts} alerts -> {RESULTS / 'alerts.jsonl'}")


if __name__ == "__main__":
    main()
