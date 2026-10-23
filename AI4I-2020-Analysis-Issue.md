# Issue: AI4I 2020 Predictive Maintenance Dataset — Analysis Study

## Overview
Extend the `26-ai-new-power` repository with a second dataset study: the UCI AI4I 2020 Predictive Maintenance dataset (10,000 synthetic milling-process records). Scope is decided upfront: LogReg baseline + Keras MLP with class-weight imbalance handling (ROC/AUC + 5-fold CV), dual prediction targets (binary Machine failure and 5-class failure mode). The pipeline follows the existing Iris study's TDD + one-command reproducibility conventions.

## Dataset
- Source: UCI ML Repository #601 (https://archive.ics.uci.edu/dataset/601/ai4i), DOI 10.24432/C5HS5C
- 10,000 rows × 14 columns, ~510 KB, no missing values
- Features: `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`; `Product ID` and `UID` are identifiers and excluded
- Targets: `Machine failure` (binary, 3.39% positive — imbalanced) and 5 one-hot failure modes: `TWF` (46), `HDF` (115), `PWF` (95), `OSF` (98), `RNF` (19)
- Citation: Matzka, S. "Explainable Artificial Intelligence for Predictive Maintenance Applications." AI4I 2020, pp. 69–74.

## Tasks

### 1. Dataset Acquisition and Storage
- [x] Download `ai4i2020.csv` via HTTPS from UCI and save as `data/ai4i2020.csv`
- [x] Add `data/AI4I-README.md` data dictionary (fields, units, class balance, citation)
- [x] TDD tests: file exists, 10,000×14 shape, expected columns, failure-rate sanity range

### 2. EDA and 3D PCA Visualization
- [x] `src/ai4i_eda.py` text summary: shape, dtypes, missing, duplicates, `Type`/failure-mode counts, tool-wear stats by mode, correlations
- [x] Figures in `images/`: tool-wear and torque vs UDI plots, PCA 3-D feature-space scatter (8-D→3 components, colored by failure mode) → `images/ai4i_pca_3d.png`

### 3. Classification Models (Machine failure + failure mode)
- [x] 80/20 stratified train-test split, `StandardScaler` fit on train only, fixed seed 42
- [x] Baseline: Logistic Regression (`class_weight="balanced"`)
- [x] Main: Keras MLP (8 → 32 ReLU → 16 ReLU → softmax), Adam, `sparse_categorical_crossentropy` with inverse-frequency class weights
- [x] Task A: binary `Machine failure` — report ROC/AUC, F1, confusion matrix
- [x] Task B: 6-class failure mode (label = dominant active mode, 0 = no failure) — report accuracy, per-mode F1, confusion matrix
- [x] 5-fold stratified CV for both tasks
- [x] Figures: `images/ai4i_training_history.png`, `images/ai4i_confusion_matrix.png`, `images/ai4i_roc_curve.png`
- [x] `results/ai4i_metrics.json`, `results/ai4i_classification_report.txt`

### 4. Mini Paper
- [x] `docs/ai4i_nn_analysis.md` (~900 words): Abstract, Introduction, Dataset, Methodology (both tasks, imbalance strategy), Results (AUC/F1 numbers), Discussion (failure-mode confusion patterns, tool-wear signal), Conclusion, References

## TDD Conventions (carry over from Iris study)
- [x] Tests first: 4 task files under `tests/` (`test_ai4i_task1_data.py` … `test_ai4i_task4_paper.py`)
- [x] Red → Green → Refactor; 21 AI4I tests total; all pass from repo root
- [x] One-command pipeline: `python src/ai4i_run.py` (or `python src/run_all.py --dataset ai4i`)
- [x] Seed 42 everywhere for reproducibility

## Acceptance Criteria
- [x] `data/ai4i2020.csv` committed with data dictionary
- [x] `images/ai4i_pca_3d.png` + tool-wear + ROC figures present and non-trivial in size
- [x] Both tasks beat the majority-class baseline by a clear margin; AUC and 5-fold CV reported
- [x] All new pytest tests pass; existing 15 Iris tests still pass (36 total tests: 15 Iris + 21 AI4I)
- [x] Mini paper reports the actual measured results and references the figures

## Measured Results
| Metric | Value |
| --- | ---: |
| Binary machine-failure — LogReg baseline | AUC 0.9069, F1 0.2424 |
| Binary machine-failure — Keras MLP | **AUC 0.9660, F1 0.4938**, best epoch 113 |
| 6-class failure mode — Keras MLP | **macro F1 0.4366** (HDF 0.565, PWF 0.545, OSF 0.473, TWF 0.110, RNF 0.000) |
| 5-fold CV — binary AUC | 0.9003 ± 0.0089 |
| 5-fold CV — mode macro F1 | 0.3518 ± 0.0088 |
| PCA 3-component explained variance | 23.9% + 23.0% + 22.8% = 69.8% |
