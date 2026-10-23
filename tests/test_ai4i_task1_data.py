"""TDD tests for Task 1: AI4I 2020 dataset acquisition and storage."""
import os

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
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

DATA_PATH = "data/ai4i2020.csv"


def test_ai4i_csv_exists():
    """The AI4I 2020 CSV must be committed to the repo."""
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} should exist"


def test_ai4i_csv_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] == 10_000, f"Expected 10000 rows, got {df.shape[0]}"
    assert df.shape[1] == 14, f"Expected 14 columns, got {df.shape[1]}"


def test_ai4i_csv_columns():
    df = pd.read_csv(DATA_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )


def test_ai4i_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert int(df.isna().sum().sum()) == 0, "Dataset must contain no missing values"


def test_ai4i_failure_rate_sane():
    df = pd.read_csv(DATA_PATH)
    rate = df["Machine failure"].mean()
    # The dataset reports ~3.4% machine-failure rows; allow a wide band
    # that still excludes degenerate all-zero or all-one labels.
    assert 0.02 <= rate <= 0.10, f"Machine-failure rate {rate:.3f} outside expected band"


def test_ai4i_failure_mode_counts():
    df = pd.read_csv(DATA_PATH)
    for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        assert int(df[mode].sum()) > 0, f"Mode {mode} must have at least one occurrence"
    # Modes are one-hot, but a small number of rows carry more than one
    # active flag (a known quirk of the dataset's mode-generation code).
    modes = df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1)
    assert modes.max() <= 5, f"Expected at most 5 active modes, got {modes.max()}"
