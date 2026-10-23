"""Exploratory data analysis for the AI4I 2020 dataset."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ai4i_config import (
    AI4I_EDA_SUMMARY_PATH,
    AI4I_DATA_PATH,
    AI4I_PCA_PATH,
    FEATURES_AI4I,
    IMAGES_DIR,
    RESULTS_DIR,
    FAILURE_MODE_CODES,
)


def _feature_columns() -> list[str]:
    return [c for c in FEATURES_AI4I if c != "Type"]


def summarize(frame: pd.DataFrame, path=AI4I_EDA_SUMMARY_PATH) -> str:
    """Write a plain-text EDA summary and return the same string."""
    lines: list[str] = []
    lines.append("AI4I 2020 PREDICTIVE MAINTENANCE DATASET - EDA")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Source file: {AI4I_DATA_PATH.name}")
    lines.append(f"Rows: {len(frame)}   Columns: {len(frame.columns)}")
    lines.append("")
    lines.append("Columns and data types")
    lines.append("-" * 50)
    for column in frame.columns:
        lines.append(f"  {column:<30} {str(frame[column].dtype)}")
    lines.append("")
    lines.append("Missing values per column")
    lines.append("-" * 50)
    lines.append(frame.isna().sum().to_string())
    lines.append("")
    lines.append(f"Duplicate rows: {int(frame.duplicated().sum())}")
    lines.append("")
    lines.append("Type (product size class) distribution")
    lines.append("-" * 50)
    counts = frame["Type"].value_counts()
    for name, count in counts.items():
        lines.append(f"  {name:<5} {count:>6}  ({count / len(frame):.1%})")
    lines.append("")
    lines.append("Machine-failure and failure-mode counts")
    lines.append("-" * 50)
    for mode in FAILURE_MODE_CODES:
        if mode == "none":
            value = int((frame["Machine failure"] == 0).sum())
        else:
            value = int(frame[mode].sum())
        lines.append(f"  {mode:<10} {value:>6}  ({value / len(frame):.2%})")
    lines.append("")
    lines.append("Descriptive statistics of numeric features")
    lines.append("-" * 50)
    numeric = [c for c in frame.columns if frame[c].dtype != "object"]
    lines.append(frame[numeric].describe().T.round(3).to_string())
    lines.append("")
    lines.append("Pearson correlation between numeric features")
    lines.append("-" * 50)
    lines.append(frame[numeric].corr().round(3).to_string())
    lines.append("")
    lines.append("Mean tool wear and torque by product type")
    lines.append("-" * 50)
    lines.append(
        frame.groupby("Type")
        .agg({
            "Tool wear [min]": "mean",
            "Torque [Nm]": "mean",
            "Process temperature [K]": "mean",
        })
        .round(3)
        .to_string()
    )
    lines.append("")

    text = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def plot_tool_wear(frame: pd.DataFrame,
                   path=IMAGES_DIR / "ai4i_tool_wear.png") -> None:
    """Tool wear and torque vs row order, colored by failure-mode label."""
    mode_cols = FAILURE_MODE_CODES[1:]
    labels = ["none"] + mode_cols

    def _label(row) -> str:
        active = [m for m in mode_cols if row[m] == 1]
        return active[0] if active else "none"

    df = frame.copy()
    df["mode"] = df.apply(_label, axis=1)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for label, color in zip(
        labels,
        ["#999999", "#4C72B0", "#DD8452", "#55A868", "#B0568A", "#6A8E9E"],
    ):
        mask = df["mode"] == label
        if not mask.any():
            continue
        axes[0].scatter(
            df.loc[mask, "UDI"],
            df.loc[mask, "Tool wear [min]"],
            s=12,
            alpha=0.45,
            color=color,
            label=f"{label} (n={int(mask.sum())})",
        )
    axes[0].set_xlabel("UDI (row order)", fontsize=11)
    axes[0].set_ylabel("Tool wear [min]", fontsize=11)
    axes[0].set_title("Tool wear by failure mode", fontsize=13)
    axes[0].legend(fontsize=9, loc="upper left")

    for label, color in zip(
        ["none"] + ["TWF", "HDF", "PWF", "OSF", "RNF"],
        ["#999999", "#4C72B0", "#DD8452", "#55A868", "#B0568A", "#6A8E9E"],
    ):
        mask = df["mode"] == label
        if not mask.any():
            continue
        axes[1].scatter(
            df.loc[mask, "UDI"],
            df.loc[mask, "Torque [Nm]"],
            s=12,
            alpha=0.45,
            color=color,
            label=f"{label} (n={int(mask.sum())})",
        )
    axes[1].set_xlabel("UDI (row order)", fontsize=11)
    axes[1].set_ylabel("Torque [Nm]", fontsize=11)
    axes[1].set_title("Torque by failure mode", fontsize=13)
    axes[1].legend(fontsize=9, loc="upper left")

    fig.suptitle("AI4I 2020: tool wear and torque vs row order", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_eda(frame: pd.DataFrame) -> dict:
    """Run every EDA step and return a small JSON-friendly summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(frame)
    plot_tool_wear(frame)

    mode_cols = FAILURE_MODE_CODES[1:]
    counts = {label: int((frame["Machine failure"] == 0).sum()) if label == "none"
              else int(frame[label].sum())
              for label in FAILURE_MODE_CODES}

    return {
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "failure_mode_counts": counts,
        "type_counts": frame["Type"].value_counts().to_dict(),
    }
