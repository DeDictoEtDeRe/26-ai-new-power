"""LogReg baseline and Keras MLP training for AI4I 2020 tasks."""
from __future__ import annotations

import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as tf_keras

from ai4i_config import (
    AI4I_DATA_PATH,
    AI4I_HISTORY_PATH,
    AI4I_METRICS_PATH,
    AI4I_MODEL_PATH,
    AI4I_REPORT_PATH,
    BATCH_SIZE,
    FEATURES_AI4I,
    FAILURE_MODE_CODES,
    IMAGES_DIR,
    MLP_EPOCHS,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)
from ai4i_data import split_ai4i


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Best-effort reproducible seeding."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf_keras.utils.set_random_seed(seed)


def build_mlp(
    input_dim: int,
    n_classes: int,
    hidden_units: tuple[int, ...] = (32, 16),
) -> keras.Model:
    """Compact feed-forward classifier for the AI4I tasks."""
    layers = [keras.Input(shape=(input_dim,), name="ai4i_features")]
    for i, units in enumerate(hidden_units, start=1):
        layers.append(
            keras.layers.Dense(units, activation="relu", name=f"hidden_{i}")
        )
    layers.append(
        keras.layers.Dense(n_classes, activation="softmax", name="predict")
    )
    model = keras.Sequential(layers, name="ai4i_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    class_weights: dict | None = None,
    epochs: int = MLP_EPOCHS,
) -> tuple[keras.Model, "keras.callbacks.History"]:
    set_seeds()
    model = build_mlp(X_train.shape[1], n_classes)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=0,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=True,
        verbose=0,
    )
    return model, history


def _class_weights_from_array(y: np.ndarray, n_classes: int) -> dict:
    """Inverse-frequency class weights returned as a Keras-compatible dict."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    total = float(counts.sum())
    weights = total / (n_classes * counts)
    return {int(idx): float(w) for idx, w in enumerate(weights)}


def binary_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Logistic-regression baseline with balanced class weighting."""
    logreg = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    logreg.fit(X_train, y_train)
    proba = logreg.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
    }


def binary_mlp(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, "keras.callbacks.History", keras.Model]:
    """Binary machine-failure Keras MLP with class weights."""
    n_classes = 2
    weights = _class_weights_from_array(y_train, n_classes)
    model, history = _fit_mlp(X_train, y_train, X_val, y_val, n_classes,
                              class_weights=weights)
    proba = model.predict(X_test, verbose=0)
    pred = proba.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba[:, 1])),
    }
    return metrics, history, model


def failure_mode_mlp(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, "keras.callbacks.History", keras.Model]:
    """6-class failure-mode Keras MLP."""
    n_classes = len(FAILURE_MODE_CODES)
    weights = _class_weights_from_array(y_train, n_classes)
    model, history = _fit_mlp(X_train, y_train, X_val, y_val, n_classes,
                              class_weights=weights)
    proba = model.predict(X_test, verbose=0)
    pred = proba.argmax(axis=1)
    per_class = precision_recall_fscore_support(
        y_test, pred, labels=list(range(n_classes)), zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_test, pred, labels=list(range(n_classes)), average="macro",
        zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "f1_per_mode": {
            name: float(f1_v)
            for name, f1_v in zip(FAILURE_MODE_CODES, per_class[2])
        },
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=list(range(n_classes))
        ).tolist(),
        "y_pred": [int(v) for v in pred],
    }, history, model


def cross_validate(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_mode: np.ndarray,
    folds: int = 5,
) -> dict:
    """Stratified 5-fold CV for both tasks; returns AUCs and F1s."""
    binary_folds: list[float] = []
    mode_folds: list[float] = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True,
                                random_state=RANDOM_STATE)
    for train_idx, valid_idx in splitter.split(X, y_binary):
        X_tr, X_va = X[train_idx], X[valid_idx]
        yb_tr, yb_va = y_binary[train_idx], y_binary[valid_idx]
        ym_tr, ym_va = y_mode[train_idx], y_mode[valid_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)

        # Binary: LogReg with balanced weights (cheap, robust)
        logreg = LogisticRegression(max_iter=3000, class_weight="balanced",
                                    random_state=RANDOM_STATE)
        logreg.fit(X_tr_s, yb_tr)
        auc_fold = float(roc_auc_score(yb_va, logreg.predict_proba(X_va_s)[:, 1]))
        binary_folds.append(auc_fold)

        # 6-class failure mode: train a fresh small MLP (fewer epochs)
        weights = _class_weights_from_array(ym_tr, 6)
        model, _ = _fit_mlp(X_tr_s, ym_tr, X_va_s, ym_va, 6,
                            class_weights=weights, epochs=40)
        pred = model.predict(X_va_s, verbose=0).argmax(axis=1)
        f1_fold = float(f1_score(ym_va, pred, average="macro", zero_division=0))
        mode_folds.append(f1_fold)

    return {
        "folds": folds,
        "binary_folds": binary_folds,
        "binary_auc_mean": float(np.mean(binary_folds)),
        "binary_auc_std": float(np.std(binary_folds)),
        "mode_folds": mode_folds,
        "mode_f1_mean": float(np.mean(mode_folds)),
        "mode_f1_std": float(np.std(mode_folds)),
    }


def plot_training_history(
    history: "keras.callbacks.History",
    best_epoch: int,
    path=IMAGES_DIR / "ai4i_training_history.png",
) -> None:
    """Training/val loss and accuracy curves for the binary MLP."""
    records = history.history
    epochs = range(1, len(records["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(epochs, records["accuracy"], label="Train", color="#4C72B0", lw=2)
    axes[0].plot(epochs, records["val_accuracy"], label="Val", color="#DD8452", lw=2)
    axes[0].axvline(best_epoch, color="#888888", ls="--", lw=1.2,
                    label=f"Best ({best_epoch})")
    axes[0].set_title("Binary machine-failure: accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[1].plot(epochs, records["loss"], label="Train", color="#4C72B0", lw=2)
    axes[1].plot(epochs, records["val_loss"], label="Val", color="#DD8452", lw=2)
    axes[1].axvline(best_epoch, color="#888888", ls="--", lw=1.2,
                    label=f"Best ({best_epoch})")
    axes[1].set_title("Binary machine-failure: loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    fig.suptitle("AI4I 2020 Keras MLP training dynamics (binary task)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray,
             path=IMAGES_DIR / "ai4i_roc_curve.png") -> None:
    """ROC curve for the binary machine-failure task."""
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6.5, 5.8))
    plt.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"MLP (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    plt.fill_between(fpr, tpr, alpha=0.15, color="#4C72B0")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AI4I 2020 binary machine-failure ROC curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_confusion(
    matrix: list[list[int]],
    accuracy: float,
    labels: list[str],
    path=IMAGES_DIR / "ai4i_confusion_matrix.png",
) -> None:
    """Annotated confusion-matrix heatmap."""
    import seaborn as sns

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        square=True, linewidths=1, linecolor="white",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 11}, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"AI4I 2020 failure-mode confusion matrix (accuracy = {accuracy:.1%})",
                 fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def run() -> dict:
    """Run the full AI4I 2020 classification workflow and persist artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds()

    frame = pd.read_csv(AI4I_DATA_PATH)
    X, y_binary, y_mode = split_ai4i(frame)

    # 80/20 stratified split on the binary target (dominant task).
    X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
        X, y_binary, y_mode,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_binary,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # Hold out 20% of the training partition for early stopping.
    X_fit, X_val, yb_fit, yb_val, ym_fit, ym_val = train_test_split(
        X_train_s, yb_train, ym_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=yb_train,
    )

    # --- Binary task ---
    binary_baseline_metrics = binary_baseline(
        X_train_s, X_test_s, yb_train, yb_test
    )
    binary_mlp_metrics, binary_history, binary_model = binary_mlp(
        X_fit, X_val, yb_fit, yb_val, X_test_s, yb_test
    )
    binary_probabilities = binary_model.predict(X_test_s, verbose=0)
    best_epoch_binary = int(
        np.argmin(binary_history.history["val_loss"]) + 1
    )

    # --- Failure-mode task ---
    mode_metrics, mode_history, _mode_model = failure_mode_mlp(
        X_fit, X_val, ym_fit, ym_val, X_test_s, ym_test
    )

    # --- 5-fold cross-validation ---
    cv_result = cross_validate(X, y_binary, y_mode)

    # --- Visualizations ---
    plot_training_history(binary_history, best_epoch_binary)
    plot_roc(yb_test, binary_probabilities)
    plot_confusion(mode_metrics["confusion_matrix"], mode_metrics["accuracy"],
                   FAILURE_MODE_CODES)

    # --- Reports ---
    report_lines = [
        "AI4I 2020 - CLASSIFICATION REPORTS",
        "=" * 40,
        "",
        "## Binary task: Machine failure",
        f"LogReg baseline: F1={binary_baseline_metrics['f1']:.4f}  "
        f"AUC={binary_baseline_metrics['auc']:.4f}",
        f"Keras MLP:     F1={binary_mlp_metrics['f1']:.4f}  "
        f"AUC={binary_mlp_metrics['auc']:.4f}",
        "",
        "## 6-class task: failure mode",
    ]
    report_lines.append(classification_report(
        ym_test,
        np.array(mode_metrics["y_pred"]),
        labels=list(range(len(FAILURE_MODE_CODES))),
        target_names=FAILURE_MODE_CODES,
        zero_division=0,
    ))
    AI4I_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    payload = {
        "dataset": {
            "rows": int(len(frame)),
            "features": list(FEATURES_AI4I),
            "failure_mode_labels": list(FAILURE_MODE_CODES),
            "failure_rate": float(y_binary.mean()),
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "validation_rows": int(len(X_val)),
        },
        "binary_task": {
            "logreg_baseline": binary_baseline_metrics,
            "mlp": binary_mlp_metrics,
        },
        "failure_mode_task": {
            "mlp": mode_metrics,
        },
        "cv": cv_result,
        "best_epoch_binary": best_epoch_binary,
    }

    AI4I_HISTORY_PATH.write_text(
        json.dumps({k: [float(v) for v in vals]
                    for k, vals in binary_history.history.items()}, indent=2),
        encoding="utf-8",
    )
    AI4I_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
