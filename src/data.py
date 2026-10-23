"""Dataset acquisition and loading for the Iris study.

The canonical copy of the dataset is ``data/iris.csv``.  If it is missing,
it is fetched from scikit-learn (the same 150 records published in the UCI
Machine Learning Repository) and written to disk.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris

from config import CLASSES, DATA_PATH, FEATURES, ensure_dirs


def fetch_iris() -> pd.DataFrame:
    """Return the Iris dataset as a tidy DataFrame."""
    bunch = load_iris(as_frame=True)
    frame = bunch.frame.copy()

    # scikit-learn names the target column "target"; map it to species names.
    frame["species"] = frame["target"].map(dict(enumerate(bunch.target_names)))
    frame = frame[FEATURES + ["species"]]
    frame[FEATURES] = frame[FEATURES].astype(float)
    return frame


def save_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Fetch the dataset and persist it as CSV (idempotent)."""
    ensure_dirs()
    frame = fetch_iris()
    frame.to_csv(path, index=False, float_format="%.1f")
    return frame


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Load the local CSV, creating it from scikit-learn when absent."""
    if not path.exists():
        return save_dataset(path)

    frame = pd.read_csv(path)
    missing = [column for column in FEATURES + ["species"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def split_features_target(frame: pd.DataFrame):
    """Return ``(X, y)`` with feature matrix and integer-encoded labels."""
    X = frame[FEATURES].to_numpy(dtype=float)
    y = frame["species"].map({name: index for index, name in enumerate(CLASSES)}).to_numpy()
    return X, y
