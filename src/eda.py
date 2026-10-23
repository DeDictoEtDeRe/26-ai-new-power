"""Exploratory data analysis for the Iris dataset."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CLASSES, EDA_SUMMARY_PATH, FEATURES, IMAGES_DIR, RESULTS_DIR, ensure_dirs


def summarize(frame: pd.DataFrame, path=EDA_SUMMARY_PATH) -> str:
    """Write a plain-text EDA summary and return it."""
    lines: list[str] = []
    lines.append("IRIS DATASET - EXPLORATORY DATA ANALYSIS")
    lines.append("=" * 45)
    lines.append("")
    lines.append(f"Source file: data/iris.csv")
    lines.append(f"Rows: {len(frame)}   Columns: {len(frame.columns)}")
    lines.append("")
    lines.append("Columns and data types")
    lines.append("-" * 45)
    for column in frame.columns:
        lines.append(f"  {column:<22} {str(frame[column].dtype)}")
    lines.append("")
    lines.append("First five rows")
    lines.append("-" * 45)
    lines.append(frame.head().to_string(index=False))
    lines.append("")
    lines.append("Missing values per column")
    lines.append("-" * 45)
    lines.append(frame.isna().sum().to_string())
    lines.append("")
    lines.append(f"Duplicate rows: {int(frame.duplicated().sum())}")
    lines.append("")
    lines.append("Class distribution")
    lines.append("-" * 45)
    counts = frame["species"].value_counts().reindex(CLASSES)
    for name, count in counts.items():
        lines.append(f"  {name:<12} {count:>3}  ({count / len(frame):.1%})")
    lines.append("")
    lines.append("Descriptive statistics (cm)")
    lines.append("-" * 45)
    lines.append(frame[FEATURES].describe().round(3).to_string())
    lines.append("")
    lines.append("Pearson correlation between features")
    lines.append("-" * 45)
    lines.append(frame[FEATURES].corr().round(3).to_string())
    lines.append("")
    lines.append("Mean feature values by species (cm)")
    lines.append("-" * 45)
    lines.append(frame.groupby("species")[FEATURES].mean().reindex(CLASSES).round(3).to_string())
    lines.append("")

    text = "\n".join(lines)
    ensure_dirs()
    path.write_text(text, encoding="utf-8")
    return text


def plot_pairplot(frame: pd.DataFrame, path=IMAGES_DIR / "iris_pairplot.png") -> None:
    """Save a 4x4 scatterplot matrix colored by species."""
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.pairplot(
        frame,
        hue="species",
        hue_order=CLASSES,
        vars=FEATURES,
        corner=True,
        diag_kind="kde",
        plot_kws={"s": 32, "alpha": 0.8, "edgecolor": "white", "linewidth": 0.4},
        diag_kws={"fill": False, "linewidth": 2},
        height=2.4,
    )
    grid.figure.suptitle("Iris feature relationships by species", y=1.02, fontsize=15)
    grid.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(grid.figure)


def plot_correlation(frame: pd.DataFrame, path=IMAGES_DIR / "iris_correlation.png") -> None:
    """Save a correlation heatmap for the four measurements."""
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = frame[FEATURES].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature correlation matrix (Pearson)", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_distributions(frame: pd.DataFrame, path=IMAGES_DIR / "iris_feature_distributions.png") -> None:
    """Save boxplots of each feature grouped by species."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    palette = {"Iris setosa": "#4C72B0", "Iris versicolor": "#DD8452", "Iris virginica": "#55A868"}
    for ax, feature in zip(axes.ravel(), FEATURES):
        sns.boxplot(
            data=frame,
            x="species",
            y=feature,
            hue="species",
            order=CLASSES,
            hue_order=CLASSES,
            palette=palette,
            width=0.55,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=frame,
            x="species",
            y=feature,
            order=CLASSES,
            color="black",
            alpha=0.45,
            size=3,
            ax=ax,
        )
        ax.set_title(feature, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("cm")
    fig.suptitle("Measurement distributions by species", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_eda(frame: pd.DataFrame) -> dict:
    """Run every EDA step and return a small JSON-friendly summary."""
    ensure_dirs()
    summary = summarize(frame)
    plot_pairplot(frame)
    plot_correlation(frame)
    plot_distributions(frame)

    counts = frame["species"].value_counts().reindex(CLASSES)
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {name: int(count) for name, count in counts.items()},
        "summary_path": str(EDA_SUMMARY_PATH.relative_to(RESULTS_DIR.parent)),
        "summary_chars": len(summary),
    }
