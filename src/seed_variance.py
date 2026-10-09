"""Retrain both ML models with several seeds and report the spread of their test metrics.

python src/seed_variance.py --seeds 0 1 2  ->  results/<source>/seed_variance.csv
The rule baseline has no randomness, so it is not repeated.
"""
import argparse
import pickle

import pandas as pd

import model_transformer
import train
from evaluate import operating_point, score_and_threshold
from features import RESULTS, load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    ev, split = load()
    days = (split["test_end"] - split["calib_end"]) / 86400
    rows = []
    for seed in args.seeds:
        if seed == 0 and (RESULTS / "transformer.pt").exists():
            with open(RESULTS / "iforest.pkl", "rb") as f:
                iforest = pickle.load(f)
            bundle = model_transformer.load(RESULTS / "transformer.pt")
        else:
            iforest = train.fit_iforest(ev, split, seed)
            bundle = train.fit_transformer(ev, split, args.epochs, seed)
        test, thresholds, _ = score_and_threshold(ev.copy(), split, iforest, bundle)
        y = test.label.to_numpy(bool)
        for m in ("iforest", "transformer"):
            op = operating_point(y, test[f"score_{m}"].to_numpy(), thresholds[m], days)
            rows.append(dict(seed=seed, model=m, **{k: op[k] for k in ("pr_auc", "precision", "recall", "fp_per_day")}))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "seed_variance.csv", index=False)
    print(df.groupby("model")[["pr_auc", "precision", "recall", "fp_per_day"]].agg(["mean", "std", "min", "max"]).round(4).T)


if __name__ == "__main__":
    main()
