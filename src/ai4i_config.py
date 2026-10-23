"""Shared constants for the AI4I 2020 study (no circular import)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

AI4I_DATA_PATH = DATA_DIR / "ai4i2020.csv"
AI4I_EDA_SUMMARY_PATH = RESULTS_DIR / "ai4i_eda_summary.txt"
AI4I_PCA_PATH = RESULTS_DIR / "ai4i_pca.json"
AI4I_METRICS_PATH = RESULTS_DIR / "ai4i_metrics.json"
AI4I_REPORT_PATH = RESULTS_DIR / "ai4i_classification_report.txt"
AI4I_HISTORY_PATH = RESULTS_DIR / "ai4i_history.json"
AI4I_MODEL_PATH = RESULTS_DIR / "ai4i_mlp.keras"
AI4I_PAPER_PATH = DOCS_DIR / "ai4i_nn_analysis.md"

# Numeric features used for modeling.
# `Type` is one-hot encoded by `src/ai4i_data.py:load_ai4i`.
FEATURES_AI4I = [
    "Type_L",
    "Type_M",
    "Type_H",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Ground-truth labels: `Machine failure` (binary) plus 6-class failure mode
# (`none` + 5 modes), derived from the one-hot columns TWF/HDF/PWF/OSF/RNF.
FAILURE_MODE_CODES = ["none", "TWF", "HDF", "PWF", "OSF", "RNF"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
MLP_EPOCHS = 120
BATCH_SIZE = 64
