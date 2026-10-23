"""Principal Component Analysis and the 3D feature-space figure."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import CLASSES, FEATURES, IMAGES_DIR, PCA_PATH, ensure_dirs


def fit_pca(X: np.ndarray, n_components: int = 3):
    """Standardize features, then fit PCA and return (transformed, pca, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, scaler


def plot_pca_3d(
    X_pca: np.ndarray,
    y: np.ndarray,
    explained: np.ndarray,
    path=IMAGES_DIR / "iris_pca_3d.png",
) -> None:
    """Save a 3D scatter plot of the first three principal components."""
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    markers = ["o", "s", "^"]

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    for index, (name, color, marker) in enumerate(zip(CLASSES, colors, markers)):
        mask = y == index
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            X_pca[mask, 2],
            s=52,
            alpha=0.85,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.5,
            label=f"{name} (n={int(mask.sum())})",
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)", fontsize=11, labelpad=10)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)", fontsize=11, labelpad=10)
    ax.set_zlabel(f"PC3 ({explained[2]:.1%} variance)", fontsize=11, labelpad=10)
    ax.view_init(elev=22, azim=45)
    ax.set_title(
        "Iris dataset: 3D PCA projection of the four standardized measurements",
        fontsize=14,
        pad=18,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5,
        0.02,
        f"First three components explain {explained.sum():.1%} of total variance "
        "(features standardized before PCA).",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_pca(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit PCA, save the figure and the explained-variance record."""
    ensure_dirs()
    X_pca, pca, _ = fit_pca(X, n_components=3)
    explained = pca.explained_variance_ratio_
    plot_pca_3d(X_pca, y, explained)

    payload = {
        "n_components": 3,
        "explained_variance_ratio": [float(value) for value in explained],
        "cumulative_variance": [float(value) for value in np.cumsum(explained)],
        "loadings": {
            feature: [float(value) for value in row]
            for feature, row in zip(FEATURES, pca.components_)
        },
        "figure": "images/iris_pca_3d.png",
    }
    PCA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
