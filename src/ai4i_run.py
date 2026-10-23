"""Entry point for the AI4I 2020 study pipeline.

Run from the repository root:
    python src/run_all.py --dataset ai4i
"""
from __future__ import annotations

import argparse

from ai4i_data import load_ai4i, split_ai4i
from ai4i_eda import run_eda
from ai4i_pca_plot import run_pca
from ai4i_train import run as run_train


def main(refresh_data: bool = False, skip_cv: bool = False) -> dict:
    frame = load_ai4i()
    print(f"AI4I 2020 dataset: {len(frame)} rows x {len(frame.columns)} columns")
    print("Running EDA ...")
    eda_info = run_eda(frame)
    print("Fitting PCA and saving 3-D visualization ...")
    X, y_binary, y_mode = split_ai4i(frame)
    pca_info = run_pca(X, y_mode)
    print("Training LogReg baseline and Keras MLP ...")
    results = run_train()
    print(
        f"Binary task — LogReg AUC={results['binary_task']['logreg_baseline']['auc']:.4f}, "
        f"MLP AUC={results['binary_task']['mlp']['auc']:.4f}, "
        f"MLP F1={results['binary_task']['mlp']['f1']:.4f}"
    )
    print(
        f"Failure-mode task — MLP macro-F1={results['failure_mode_task']['mlp']['f1_macro']:.4f}"
    )
    cv = results["cv"]
    print(
        f"5-fold CV — binary AUC mean={cv['binary_auc_mean']:.4f} "
        f"(std {cv['binary_auc_std']:.4f}); mode macro-F1 mean={cv['mode_f1_mean']:.4f} "
        f"(std {cv['mode_f1_std']:.4f})"
    )
    print("Artifacts written to images/, results/.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4I 2020 study pipeline.")
    parser.add_argument("--refresh-data", action="store_true",
                        help="re-download data/ai4i2020.csv from UCI")
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip the 5-fold cross-validation estimate")
    args = parser.parse_args()
    main(refresh_data=args.refresh_data, skip_cv=args.skip_cv)
