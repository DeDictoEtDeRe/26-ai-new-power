# 26-ai-new-power — Iris Dataset Study

A complete, reproducible machine-learning study of the classic Iris dataset:
data acquisition, exploratory analysis, PCA-based 3-D visualization,
neural-network classification, and a short paper on the results.

## Dataset

`data/iris.csv` (see `data/README.md` for the full data dictionary):
150 flower measurements, 4 features in centimetres
(sepal length/width, petal length/width), 3 balanced classes
(`Iris setosa`, `Iris versicolor`, `Iris virginica` — 50 each). Refresh it at any time
with `python src/run_all.py --refresh-data`.

## Quick start

```bash
pip install -r requirements.txt

# Full pipeline: dataset → EDA → PCA plot → train/evaluate → artifacts
python src/run_all.py
```

Final printed results (seeded, reproducible run):

| Metric | Value |
| --- | ---: |
| Test accuracy (80/20 split) | 0.9667 |
| Macro F1 | 0.9666 |
| 5-fold CV accuracy | 0.9600 ± 0.0533 |

## Layout

| Path | Contents |
| --- | --- |
| `data/iris.csv`, `data/README.md` | Dataset + documentation |
| `src/` | Pipeline: `config.py`, `data.py`, `eda.py`, `pca_plot.py`, `train.py`, `run_all.py` |
| `images/` | `iris_pca_3d.png`, `iris_pairplot.png`, `iris_correlation.png`, `iris_feature_distributions.png`, `iris_training_history.png`, `iris_confusion_matrix.png`, `iris_classification_report.png` |
| `models/iris_nn.keras` | Saved Keras model |
| `results/` | `eda_summary.txt`, `pca.json`, `metrics.json`, `history.json`, `classification_report.txt`, `model_summary.txt`, `summary.json` |
| `docs/iris_nn_analysis.md` | Short paper on the experimental results |
| `tests/` | pytest suite (15 tests across the four tasks) |
| `IRISS-Analysis-Issue.md` | Project issue tracking this work |

## Tests

```bash
python -m pytest tests/ -q
```
