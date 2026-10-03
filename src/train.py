"""Train one model on the training period: python src/train.py --model {iforest,transformer}"""
import argparse
import pickle

import pandas as pd
from sklearn.ensemble import IsolationForest

import model_iforest
import model_transformer
from features import RESULTS, build_windows, load, period


def fit_iforest(ev: pd.DataFrame, split: dict, seed: int = 0) -> IsolationForest:
    """Fit on train-period windows. Burn-in windows are excluded: early in the log every host looks new."""
    windows = build_windows(ev)
    train_w = windows[period(windows, split) == "train"]
    print(f"training Isolation Forest on {len(train_w):,} windows")
    return model_iforest.fit(train_w, seed=seed)


def fit_transformer(ev: pd.DataFrame, split: dict, epochs: int = 20, seed: int = 0) -> dict:
    """Fit on all events before train_end; early-stop on the calibration period (unlabeled)."""
    train_ev = ev[ev.time < split["train_end"]]
    val_ev = ev[(ev.time >= split["train_end"]) & (ev.time < split["calib_end"])]
    print(f"training Transformer on {len(train_ev):,} events, validating on {len(val_ev):,}")
    return model_transformer.train(train_ev, val_ev, epochs=epochs, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["iforest", "transformer"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20, help="transformer: max epochs (early stopping picks the best)")
    args = ap.parse_args()
    ev, split = load()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.model == "iforest":
        with open(RESULTS / "iforest.pkl", "wb") as f:
            pickle.dump(fit_iforest(ev, split, args.seed), f)
    else:
        model_transformer.save(fit_transformer(ev, split, args.epochs, args.seed), RESULTS / "transformer.pt")
    print(f"saved {args.model} to {RESULTS}")


if __name__ == "__main__":
    main()
