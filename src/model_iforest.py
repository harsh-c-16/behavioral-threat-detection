"""Isolation Forest on per-(user, hour) window features."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import FEATURES


def matrix(windows: pd.DataFrame) -> np.ndarray:
    # Counts are heavy-tailed; log1p keeps one busy admin hour from dominating the splits.
    return np.log1p(windows[FEATURES].to_numpy(dtype=float))


def fit(train_windows: pd.DataFrame, seed: int = 0) -> IsolationForest:
    model = IsolationForest(n_estimators=300, max_samples=4096, random_state=seed, n_jobs=-1)
    return model.fit(matrix(train_windows))


def score(model: IsolationForest, windows: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous."""
    return -model.score_samples(matrix(windows))
