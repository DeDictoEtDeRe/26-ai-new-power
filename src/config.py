"""Shared paths and constants for the Iris study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
IMAGES_DIR = ROOT / "images"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

DATA_PATH = DATA_DIR / "iris.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"
HISTORY_PATH = RESULTS_DIR / "history.json"
EDA_SUMMARY_PATH = RESULTS_DIR / "eda_summary.txt"
PCA_PATH = RESULTS_DIR / "pca.json"
MODEL_PATH = MODELS_DIR / "iris_nn.keras"

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASSES = ["Iris setosa", "Iris versicolor", "Iris virginica"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 300
BATCH_SIZE = 16


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in (DATA_DIR, DOCS_DIR, IMAGES_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
