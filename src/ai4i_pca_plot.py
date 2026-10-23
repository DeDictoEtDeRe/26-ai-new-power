"""3-D PCA feature-space visualization for AI4I 2020."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ai4i_config import AI4I_PCA_PATH, FEATURES_AI4I, FAILURE_MODE_CODES, IMAGES_DIR


def fit_pca(X: np.ndarray, n_components: int = 3):
    """Standardize, then fit PCA. Returns (transformed, pca, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, scaler


def plot_pca_3d(
    X_pca: np.ndarray,
    y_mode: np.ndarray,
    explained: np.ndarray,
    path=IMAGES_DIR / "ai4i_pca_3d.png",
) -> None:
    """Save a 3-D PCA scatter colored by failure mode."""
    colors = {
        "none": "#999999",
        "TWF": "#4C72B0",
        "HDF": "#DD8452",
        "PWF": "#55A868",
        "OSF": "#B0568A",
        "RNF": "#6A8E9E",
    }
    markers = {"none": "o", "TWF": "o", "HDF": "s", "PWF": "^", "OSF": "D", "RNF": "v"}

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for code, name in enumerate(FAILURE_MODE_CODES):
        mask = y_mode == code
        if not mask.any():
            continue
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            X_pca[mask, 2],
            s=14,
            alpha=0.35,
            color=colors[name],
            marker=markers[name],
            edgecolor="none",
            label=f"{name} (n={int(mask.sum())})",
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=11, labelpad=10)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=11, labelpad=10)
    ax.set_zlabel(f"PC3 ({explained[2]:.1%})", fontsize=11, labelpad=10)
    ax.view_init(elev=22, azim=45)
    ax.set_title(
        "AI4I 2020: 3-D PCA projection of standardized features, colored by failure mode",
        fontsize=13,
        pad=16,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.text(
        0.5,
        0.02,
        f"First three components explain {explained.sum():.1%} of total variance.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_pca(X: np.ndarray, y_mode: np.ndarray) -> dict:
    """Fit PCA, save the figure and the explained-variance record."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AI4I_PCA_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_pca, pca, _ = fit_pca(X, n_components=3)
    explained = pca.explained_variance_ratio_
    plot_pca_3d(X_pca, y_mode, explained)

    payload = {
        "n_components": 3,
        "explained_variance_ratio": [float(v) for v in explained],
        "cumulative_variance": [float(v) for v in np.cumsum(explained)],
        "figure": "images/ai4i_pca_3d.png",
    }
    AI4I_PCA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
