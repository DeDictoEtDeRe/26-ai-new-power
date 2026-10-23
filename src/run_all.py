"""Run the complete Iris study from raw data to evaluation artifacts.

Usage (from the repository root):
    python src/run_all.py
    python src/run_all.py --skip-cv
    python src/run_all.py --refresh-data
"""
from __future__ import annotations

import argparse
import json

from config import DATA_PATH, FEATURES, RESULTS_DIR, ensure_dirs
from data import load_dataset, save_dataset, split_features_target
from eda import run_eda
from pca_plot import run_pca
from train import train_and_evaluate


def main(refresh_data: bool = False, run_cv: bool = True) -> dict:
    ensure_dirs()

    if refresh_data or not DATA_PATH.exists():
        print(f"Writing dataset to {DATA_PATH} ...")
        save_dataset()
    frame = load_dataset()
    print(f"Dataset: {len(frame)} rows x {len(frame.columns)} columns")

    print("Running exploratory data analysis ...")
    eda_info = run_eda(frame)

    X, y = split_features_target(frame)
    print("Fitting PCA and saving 3D visualization ...")
    pca_info = run_pca(X, y)

    print("Training neural network ...")
    results = train_and_evaluate(frame, X, y, run_cv=run_cv)

    summary = {
        "dataset": eda_info,
        "pca": pca_info,
        "test_accuracy": results["test"]["accuracy"],
        "f1_macro": results["test"]["f1_macro"],
        "cross_validation": results["cross_validation"],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    test = results["test"]
    print("")
    print("=" * 58)
    print("RESULTS")
    print("=" * 58)
    print(f"Test accuracy      : {test['accuracy']:.2%}")
    print(f"Macro precision    : {test['precision_macro']:.4f}")
    print(f"Macro recall       : {test['recall_macro']:.4f}")
    print(f"Macro F1           : {test['f1_macro']:.4f}")
    print(f"Test log loss      : {test['loss']:.4f}")
    print(f"Epochs trained     : {results['model']['epochs_trained']}")
    if results["cross_validation"]:
        cv = results["cross_validation"]
        print(f"5-fold CV accuracy : {cv['mean_accuracy']:.2%} +/- {cv['std_accuracy']:.2%}")
    print("")
    print("Artifacts written to images/, results/ and models/.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Iris classification study.")
    parser.add_argument("--refresh-data", action="store_true",
                        help="re-download/re-create data/iris.csv from scikit-learn")
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip the 5-fold cross-validation estimate")
    args = parser.parse_args()
    main(refresh_data=args.refresh_data, run_cv=not args.skip_cv)
