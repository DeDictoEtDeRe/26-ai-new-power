"""Acquire, save, and load the AI4I 2020 Predictive Maintenance dataset."""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from ai4i_config import FEATURES_AI4I, FAILURE_MODE_CODES, AI4I_DATA_PATH

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)


def fetch_ai4i() -> pd.DataFrame:
    """Download the AI4I 2020 CSV from the UCI Machine Learning Repository."""
    tmp_path = AI4I_DATA_PATH.with_suffix(".csv.tmp")
    urllib.request.urlretrieve(SOURCE_URL, tmp_path)
    frame = pd.read_csv(tmp_path)
    tmp_path.unlink(missing_ok=True)
    return frame


def save_dataset(path=AI4I_DATA_PATH) -> pd.DataFrame:
    """Fetch the dataset and persist it as CSV (idempotent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_ai4i()
    frame.to_csv(path, index=False)
    return frame


def load_ai4i(path=AI4I_DATA_PATH) -> pd.DataFrame:
    """Load the local CSV, downloading from UCI when absent."""
    if not path.exists():
        return save_dataset(path)
    frame = pd.read_csv(path)
    required = [
        "UDI",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def _mode_label(frame: pd.DataFrame) -> np.ndarray:
    """Return a 6-class integer label: 0 = no failure, 1..5 = failure mode."""
    codes = FAILURE_MODE_CODES
    mode_cols = codes[1:]
    values = frame[mode_cols].to_numpy()
    labels = np.where(values.sum(axis=1) == 0, 0, values.argmax(axis=1) + 1)
    return labels.astype(int)


def split_ai4i(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y_binary, y_mode)`` model-ready arrays.

    ``X`` stacks the one-hot encoded ``Type`` column with the four numeric
    process measurements.
    """
    type_onehot = pd.get_dummies(frame["Type"], prefix="Type", drop_first=False)
    numeric = frame[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
    ]
    X = np.column_stack(
        [type_onehot.to_numpy(dtype=float), numeric.to_numpy(dtype=float)]
    )
    y_binary = frame["Machine failure"].to_numpy()
    y_mode = _mode_label(frame)
    return X, y_binary, y_mode


def model_features(frame: pd.DataFrame) -> np.ndarray:
    """Column order guarantee: X must match ``FEATURES_AI4I``."""
    X, _, _ = split_ai4i(frame)
    return X
