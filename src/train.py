"""Neural-network training and evaluation for Iris classification."""
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
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as tf_keras

from config import (
    BATCH_SIZE,
    CLASSES,
    EPOCHS,
    HISTORY_PATH,
    IMAGES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    ensure_dirs,
)


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Make a best effort at reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf_keras.utils.set_random_seed(seed)


def build_model(input_dim: int = 4, hidden_units: tuple[int, ...] = (16, 8)) -> keras.Model:
    """Create the small feed-forward classifier used in the study."""
    layers = [keras.Input(shape=(input_dim,), name="measurements")]
    for index, units in enumerate(hidden_units, start=1):
        layers.append(keras.layers.Dense(units, activation="relu", name=f"hidden_{index}"))
    layers.append(keras.layers.Dense(len(CLASSES), activation="softmax", name="species"))
    model = keras.Sequential(layers, name="iris_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _fit_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = RANDOM_STATE,
    verbose: int = 0,
):
    """Fit a fresh model, stopping when validation loss stops improving."""
    set_seeds(seed)
    model = build_model(input_dim=X_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=40,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-4,
            verbose=0,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )
    return model, history


def evaluate_model(model, X_test_scaled: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute loss, accuracy, per-class metrics and the confusion matrix."""
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    probabilities = model.predict(X_test_scaled, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(CLASSES)), average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(CLASSES)), average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))

    report_text = classification_report(
        y_test,
        y_pred,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
        "confusion_matrix": matrix.tolist(),
        "y_true": [int(value) for value in y_test],
        "y_pred": [int(value) for value in y_pred],
        "probabilities": [[float(value) for value in row] for row in probabilities],
        "classification_report": report_text,
    }


def cross_validate(X: np.ndarray, y: np.ndarray, folds: int = 5) -> dict:
    """Stratified k-fold estimate of generalization accuracy."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    accuracies: list[float] = []

    for train_index, valid_index in splitter.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        scaler = StandardScaler().fit(X_train)
        model, _ = _fit_with_early_stopping(
            scaler.transform(X_train),
            y_train,
            scaler.transform(X_valid),
            y_valid,
        )
        accuracy = model.evaluate(scaler.transform(X_valid), y_valid, verbose=0)[1]
        accuracies.append(float(accuracy))

    return {
        "folds": folds,
        "fold_accuracy": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }


def plot_history(history, best_epoch: int, path=IMAGES_DIR / "iris_training_history.png") -> None:
    """Save accuracy and loss curves for the training and validation sets."""
    records = history.history
    epochs = range(1, len(records["loss"]) + 1)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(epochs, records["accuracy"], label="Training", color="#4C72B0", linewidth=2)
    axes[0].plot(epochs, records["val_accuracy"], label="Validation", color="#DD8452", linewidth=2)
    axes[0].axvline(best_epoch, color="#888888", linestyle="--", linewidth=1.2,
                    label=f"Best epoch ({best_epoch})")
    axes[0].set_title("Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.6, 1.02)
    axes[0].legend(loc="lower right")

    axes[1].plot(epochs, records["loss"], label="Training", color="#4C72B0", linewidth=2)
    axes[1].plot(epochs, records["val_loss"], label="Validation", color="#DD8452", linewidth=2)
    axes[1].axvline(best_epoch, color="#888888", linestyle="--", linewidth=1.2,
                    label=f"Best epoch ({best_epoch})")
    axes[1].set_title("Categorical cross-entropy loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")

    fig.suptitle("Neural-network training dynamics", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, accuracy: float,
                   path=IMAGES_DIR / "iris_confusion_matrix.png") -> None:
    """Save an annotated confusion-matrix heatmap."""
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        annot_kws={"size": 16},
        ax=ax,
    )
    ax.set_xlabel("Predicted species", fontsize=12)
    ax.set_ylabel("True species", fontsize=12)
    ax.set_title(f"Test-set confusion matrix (accuracy = {accuracy:.1%})", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_classification_report(metrics: dict,
                               path=IMAGES_DIR / "iris_classification_report.png") -> None:
    """Render precision/recall/F1 scores as a compact table figure."""
    rows = []
    for name in CLASSES:
        scores = metrics["per_class"][name]
        rows.append([
            name,
            f"{scores['precision']:.3f}",
            f"{scores['recall']:.3f}",
            f"{scores['f1']:.3f}",
            str(scores["support"]),
        ])
    rows.append([
        "macro avg",
        f"{metrics['precision_macro']:.3f}",
        f"{metrics['recall_macro']:.3f}",
        f"{metrics['f1_macro']:.3f}",
        str(sum(item["support"] for item in metrics["per_class"].values())),
    ])
    rows.append([
        "weighted avg",
        f"{metrics['precision_weighted']:.3f}",
        f"{metrics['recall_weighted']:.3f}",
        f"{metrics['f1_weighted']:.3f}",
        str(sum(item["support"] for item in metrics["per_class"].values())),
    ])

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Species", "Precision", "Recall", "F1-score", "Support"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.65)
    for column in range(5):
        table[(0, column)].set_facecolor("#4C72B0")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row in (4, 5):
        for column in range(5):
            table[(row, column)].set_facecolor("#EEF2F7")
    ax.set_title(
        f"Test-set classification report (accuracy = {metrics['accuracy']:.1%})",
        fontsize=14,
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def train_and_evaluate(
    frame: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    run_cv: bool = True,
) -> dict:
    """Run the full training/evaluation workflow and persist every artifact."""
    ensure_dirs()
    set_seeds()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hold out 20% of the training partition for early stopping.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    model, history = _fit_with_early_stopping(X_fit, y_fit, X_val, y_val)
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    metrics = evaluate_model(model, X_test_scaled, y_test)
    validation_accuracy = float(history.history["val_accuracy"][best_epoch - 1])

    plot_history(history, best_epoch)
    plot_confusion(np.asarray(metrics["confusion_matrix"]), metrics["accuracy"])
    plot_classification_report(metrics)

    (RESULTS_DIR / "classification_report.txt").write_text(
        metrics.pop("classification_report"), encoding="utf-8"
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (RESULTS_DIR / "model_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    cv_result = cross_validate(X, y) if run_cv else None

    payload = {
        "dataset": {
            "rows": int(len(frame)),
            "features": list(frame.columns[:-1]),
            "classes": CLASSES,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_class_counts": np.bincount(y_train, minlength=3).tolist(),
            "test_class_counts": np.bincount(y_test, minlength=3).tolist(),
            "validation_rows": int(len(X_val)),
        },
        "model": {
            "architecture": [4, 16, 8, 3],
            "hidden_activation": "relu",
            "output_activation": "softmax",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "epochs_trained": int(len(history.history["loss"])),
            "best_epoch": best_epoch,
            "early_stopping_patience": 40,
            "parameters": int(model.count_params()),
        },
        "validation_accuracy_at_best_epoch": validation_accuracy,
        "test": metrics,
        "cross_validation": cv_result,
    }

    HISTORY_PATH.write_text(
        json.dumps({key: [float(value) for value in values]
                    for key, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
