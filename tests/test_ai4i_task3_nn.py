"""TDD tests for Task 3: AI4I 2020 classification (LogReg baseline + Keras MLP)."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ai4i_data import FEATURES_AI4I, FAILURE_MODE_CODES, split_ai4i

DATA_PATH = "data/ai4i2020.csv"
METRICS_PATH = "results/ai4i_metrics.json"
REPORT_PATH = "results/ai4i_classification_report.txt"
HISTORY_PATH = "results/ai4i_history.json"


def _load_split():
    df = pd.read_csv(DATA_PATH)
    X, y_binary, y_mode_idx = split_ai4i(df)
    X_train, X_test, y_train, y_test, ym_train, ym_test = train_test_split(
        X, y_binary, y_mode_idx,
        test_size=0.2,
        random_state=42,
        stratify=y_binary,
    )
    scaler = StandardScaler().fit(X_train)
    return X, y_binary, y_mode_idx, X_train, X_test, y_train, y_test, ym_train, ym_test, scaler


def _majority_baseline_accuracy(y_test):
    base_rate = 1 - float(np.mean(y_test))
    return base_rate


def test_binary_task_beats_majority_baseline():
    """The binary machine-failure task must beat the all-normal baseline."""
    (
        X, y, ym,
        X_train, X_test, y_train, y_test,
        ym_train, ym_test, scaler,
    ) = _load_split()
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_s, y_train)
    auc = float(roc_auc_score(y_test, logreg.predict_proba(X_test_s)[:, 1]))
    f1 = float(f1_score(y_test, logreg.predict(X_test_s)))
    assert f1 > 0.05, f"LogReg F1 unexpectedly low: {f1:.3f}"
    assert auc > 0.85, f"Expected LogReg AUC > 0.85, got {auc:.3f}"


def test_five_class_task_reports_metrics():
    """The 5-class failure-mode task must report a sensible per-mode F1."""
    (
        X, y, ym,
        X_train, X_test, y_train, y_test,
        ym_train, ym_test, scaler,
    ) = _load_split()
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg.fit(X_train_s, ym_train)
    f1_macro = float(f1_score(ym_test, logreg.predict(X_test_s), average="macro"))
    # Majority class ("none") dominates (~94% of rows); a macro F1 that
    # exceeds macro F1 of the all-none baseline is required.
    assert f1_macro > 0.30, f"Expected macro F1 > 0.30, got {f1_macro:.3f}"


def test_confusion_matrix_recorded():
    with open(REPORT_PATH) as handle:
        text = handle.read()
    assert "Machine failure" in text
    assert "TWF" in text and "RNF" in text


def test_metrics_json_written():
    with open(METRICS_PATH) as handle:
        payload = json.load(handle)
    assert "binary_task" in payload, "binary_task missing from metrics.json"
    assert "failure_mode_task" in payload, "failure_mode_task missing from metrics.json"
    assert payload["binary_task"]["mlp"]["auc"] > 0.8
    assert "cv" in payload, "5-fold CV block missing"
    assert len(payload["cv"]["binary_folds"]) == 5


def test_training_history_present():
    with open(HISTORY_PATH) as handle:
        history = json.load(handle)
    assert "loss" in history and "accuracy" in history
    assert len(history["loss"]) >= 50, "MLP training should run at least 50 epochs"


def test_confusion_matrix_shape():
    (
        X, y, ym,
        X_train, X_test, y_train, y_test,
        ym_train, ym_test, scaler,
    ) = _load_split()
    X_test_s = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg.fit(scaler.transform(X_train), ym_train)
    matrix = confusion_matrix(ym_test, logreg.predict(X_test_s))
    assert matrix.shape == (6, 6), f"Confusion matrix should be 6x6, got {matrix.shape}"
