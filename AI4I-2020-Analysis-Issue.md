# Issue: AI4I 2020 Predictive Maintenance Dataset — Analysis Study

## Overview
Extend the `26-ai-new-power` repository with a second dataset study: the UCI AI4I 2020 Predictive Maintenance dataset (10,000 synthetic milling-process records). Scope is decided upfront: LogReg baseline + Keras MLP with class-weight imbalance handling (ROC/AUC + 5-fold CV), dual prediction targets (binary Machine failure and 5-class failure mode). The pipeline follows the existing Iris study's TDD + one-command reproducibility conventions.

## Dataset
- Source: UCI ML Repository #601 (https://archive.ics.uci.edu/dataset/601/ai4i), DOI 10.24432/C5HS5C
- 10,000 rows × 14 columns, ~510 KB, no missing values
- Features: `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`; `Product ID` and `UID` are identifiers and excluded
- Targets: `Machine failure` (binary, ~5.8% positive — imbalanced) and 5 one-hot failure modes: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
- Citation: Matzka, S. "Explainable Artificial Intelligence for Predictive Maintenance Applications." AI4I 2020, pp. 69–74.

## Tasks

### 1. Dataset Acquisition and Storage
- [ ] Download `ai4i2020.csv` via `ucimlrepo` (fallback: direct HTTPS URL) and save as `data/ai4i2020.csv`
- [ ] Add `data/AI4I-README.md` data dictionary (fields, units, class balance, citation)
- [ ] TDD tests: file exists, 10,000×14 shape, expected columns, failure-rate sanity range

### 2. EDA and 3D PCA Visualization
- [ ] `src/ai4i_eda.py` text summary: shape, dtypes, missing, duplicates, `Type`/failure-mode counts, tool-wear stats by mode, correlations
- [ ] Figures in `images/`: tool-wear vs UID line plot, feature distributions by failure mode, PCA 3-D feature-space scatter (4-D→3 components, colored by failure mode) → `images/ai4i_pca_3d.png`

### 3. Classification Models (Machine failure + failure mode)
- [ ] 80/20 stratified train-test split, `StandardScaler` fit on train only, fixed seed 42
- [ ] Baseline: Logistic Regression (class_weight="balanced")
- [ ] Main: Keras MLP (input 5 → 32 ReLU → 16 ReLU → 2 softmax), Adam, sparse_categorical_crossentropy with class weights
- [ ] Task A: binary `Machine failure` — report ROC/AUC, precision/recall/F1, confusion matrix
- [ ] Task B: 5-class failure mode (label = dominant active mode, 0 = no failure) — report accuracy, per-mode F1, confusion matrix
- [ ] 5-fold stratified CV for both tasks
- [ ] Figures: `images/ai4i_training_history.png`, `images/ai4i_confusion_matrix.png`, `images/ai4i_roc_curve.png`
- [ ] `results/ai4i_metrics.json`, `results/ai4i_classification_report.txt`

### 4. Mini Paper
- [ ] `docs/ai4i_nn_analysis.md` (800–1200 words): Abstract, Introduction, Dataset, Methodology (both tasks, imbalance strategy), Results (AUC/F1 numbers), Discussion (failure-mode confusion patterns, tool-wear signal), Conclusion, References

## TDD Conventions (carry over from Iris study)
- Tests first: 4 task files under `tests/` (`test_ai4i_task1_data.py` … `test_ai4i_task4_paper.py`)
- Red → Green → Refactor; 15+ tests total; all pass from repo root
- One-command pipeline: `python src/run_all.py --dataset ai4i`
- Seed 42 everywhere for reproducibility

## Acceptance Criteria
- [ ] `data/ai4i2020.csv` committed with data dictionary
- [ ] `images/ai4i_pca_3d.png` + tool-wear + ROC figures present and non-trivial in size
- [ ] Both tasks beat the majority-class baseline by a clear margin; AUC and 5-fold CV reported
- [ ] All new pytest tests pass; existing 15 Iris tests still pass
- [ ] Mini paper reports the actual measured results and references the figures
