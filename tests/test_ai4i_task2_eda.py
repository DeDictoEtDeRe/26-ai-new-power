"""TDD tests for Task 2: AI4I 2020 EDA and 3D PCA visualization."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.ai4i_data import FEATURES_AI4I, load_ai4i, split_ai4i  # noqa: F401

DATA_PATH = "data/ai4i2020.csv"
EDA_SUMMARY_PATH = "results/ai4i_eda_summary.txt"
PCA_PATH = "results/ai4i_pca.json"
PCA_PLOT_PATH = "images/ai4i_pca_3d.png"
TOOL_WEAR_PLOT_PATH = "images/ai4i_tool_wear.png"


def test_eda_summary_written():
    assert os.path.exists(EDA_SUMMARY_PATH), f"{EDA_SUMMARY_PATH} should exist"
    with open(EDA_SUMMARY_PATH) as handle:
        text = handle.read()
    for keyword in (
        "Rows: 10000",
        "failure-mode",
        "Pearson correlation",
        "Tool wear",
    ):
        assert keyword in text, f"EDA summary missing section: {keyword!r}"


def test_eda_summary_reports_missing_and_duplicates():
    with open(EDA_SUMMARY_PATH) as handle:
        text = handle.read()
    assert "Missing values" in text
    assert "Duplicate rows" in text


def test_pca_reduces_to_3_components():
    frame = load_ai4i()
    X, _, _ = split_ai4i(frame)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    assert X_pca.shape == (10_000, 3), f"Expected (10000, 3), got {X_pca.shape}"
    total_variance = float(sum(pca.explained_variance_ratio_))
    # The five AI4I features are moderately correlated, but three PCs
    # should still capture well over half the variance.
    assert total_variance > 0.5, f"Expected >50% variance, got {total_variance:.2%}"


def test_pca_record_saved():
    with open(PCA_PATH) as handle:
        record = json.load(handle)
    assert record["n_components"] == 3
    assert len(record["explained_variance_ratio"]) == 3
    assert os.path.getsize(PCA_PLOT_PATH) > 5_000, "PCA 3D plot too small"


def test_tool_wear_plot_exists():
    assert os.path.getsize(TOOL_WEAR_PLOT_PATH) > 5_000, "Tool-wear plot too small"
