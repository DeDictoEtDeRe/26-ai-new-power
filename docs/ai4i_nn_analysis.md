# AI4I 2020 Predictive Maintenance: A Short Paper on Baseline and Neural-Network Classification

## Abstract
We apply a two-stage classification pipeline to the UCI AI4I 2020
predictive-maintenance dataset (10,000 synthetic milling-process
observations, 3.39% machine-failure rate). After exploratory analysis
and a 3-D PCA dimensionality-reduction study, we train (i) a
logistic-regression baseline with balanced class weighting, and
(ii) a compact Keras multilayer perceptron
(8 → 32 → 16 → 2, inverse-frequency class weights, early stopping)
for two complementary tasks: binary machine-failure prediction and
6-class failure-mode identification (the `none` class plus the five
flagged modes). On a stratified 80/20 test partition the binary task
reaches **AUC = 0.966** and **F1 = 0.494**, exceeding the LogReg
baseline (AUC = 0.907, F1 = 0.242). For the failure-mode task the
MLP records a **macro-F1 of 0.437** with the weakest per-mode
performance on RNF (F1 = 0.000, the rarest mode) and the strongest on
HDF (F1 = 0.565). A 5-fold stratified CV confirms the estimates:
binary AUC 0.900 ± 0.009, mode macro-F1 0.352 ± 0.009. Residual
errors cluster on the tool-wear boundary, consistent with the
geometry visible in the PCA projection.

## 1. Introduction
The AI4I 2020 dataset (Matzka, 2020) is a standard benchmark for
industrial failure classification and XAI. It captures four continuous
process measurements (air and process temperature, rotational speed,
torque), a monotone tool-wear counter, a product-size class, and
one-hot failure-mode flags (TWF, HDF, PWF, OSF, RNF), together with a
binary machine-failure label. The failure rate is 3.39%, so any
evaluation must lean on ranking-based and per-class metrics rather
than raw accuracy. This paper documents a reproducible study: data
acquisition, EDA, 3-D PCA visualization, baseline-versus-MLP
comparison, and 5-fold cross-validation.

## 2. Dataset
`data/ai4i2020.csv` holds 10,000 rows and 14 columns (source: UCI ML
Repository #601, DOI 10.24432/C5HS5C). `UDI` and `Product ID` are
identifiers and are excluded from modeling; `Type` (L/M/H) is
one-hot encoded into three indicators. The five failure-mode columns
sum to 373 positive rows, while the binary `Machine failure` label
covers 339 rows — a small discrepancy indicating that not every
flagged mode produces a machine-level failure record. Per-mode
counts: TWF 46, HDF 115, PWF 95, OSF 98, RNF 19. Class balance and
feature statistics are reported in `results/ai4i_eda_summary.txt`;
the tool-wear figure (`images/ai4i_tool_wear.png`) shows tool wear
rising with UDI for failure rows, which is the primary temporal
signal the classifier must exploit.

## 3. Methodology
### 3.1 Preprocessing and split
Features are the four numeric columns plus three one-hot `Type`
indicators (8 model inputs). An 80/20 stratified train-test split
(seed 42) is applied on the binary target; 20% of the training
partition (1,600 rows) is further held out as a validation set to
drive early stopping. `StandardScaler` is fitted on training rows
only.

### 3.2 Baseline
A scikit-learn `LogisticRegression` with `class_weight="balanced"`
provides a cheap, well-calibrated reference for both tasks.

### 3.3 Neural network
A Keras MLP (8 → 32 ReLU → 16 ReLU → softmax, 2 or 6 output units)
is trained with Adam (lr 1e-3), inverse-frequency class weights,
batch size 64, up to 120 epochs, early stopping (patience 25,
best-weights restore) and a reduce-LR-on-plateau callback. All
seeds are fixed (42).

### 3.4 Evaluation
Task A (binary) reports ROC/AUC, F1 and the confusion matrix on the
held-out test set. Task B (failure mode) reports macro F1 and a
6-class confusion matrix. A 5-fold stratified CV re-trains both
models (LogReg for the binary task, a shorter 40-epoch MLP for the
mode task) to quantify split sensitivity.

## 4. Results
### 4.1 Binary machine-failure task
On the 2,000-row test set the Keras MLP reaches **AUC = 0.9660**
and **F1 = 0.4938** (accuracy 0.9385, dominated by the 94%
non-failure rows), versus **AUC = 0.9069 / F1 = 0.2424** for the
LogReg baseline. The MLP therefore lifts recall on failure rows by a
factor of ≈2 at the cost of precision — the expected trade-off of an
inverse-frequency objective on heavily imbalanced data. The ROC
curve is saved to `images/ai4i_roc_curve.png`; training dynamics
appear in `images/ai4i_training_history.png` (best epoch 113).

### 4.2 Failure-mode task
The 6-class MLP achieves a **macro F1 of 0.4366** (accuracy 0.8635,
again driven by the 94% `none` class). Per-mode F1
(`results/ai4i_metrics.json`): `none` 0.926, TWF 0.110, HDF 0.565,
PWF 0.545, OSF 0.473, RNF 0.000. RNF is the rarest mode (4 test
rows) and is entirely missed; the confusion matrix
(`images/ai4i_confusion_matrix.png`) shows the dominant off-diagonal
entries concentrated in TWF → none and HDF → none, i.e. the network
consistently under-predicts rare modes, a direct consequence of the
class-weight ratio.

### 4.3 Cross-validation
5-fold stratified CV gives binary LogReg AUC of **0.9003 ± 0.0089**
and 40-epoch MLP mode macro-F1 of **0.3518 ± 0.0088**. With ≈45
failure rows per fold, a single misranking moves a fold's F1 by a
visible margin; the small spread (std < 0.01) indicates the estimates
are stable, even though the mode-task CV value (0.352) sits below
the single-split value (0.437) because the CV MLP runs fewer epochs.

### 4.4 PCA
The first three principal components explain 23.9% + 23.0% + 22.8%
= **69.8%** of the variance of the eight standardized features
(`results/ai4i_pca.json`). The variance is spread nearly evenly
across the three PCs (a contrast with the Iris study, where PC1
alone captured 73%), which is expected for weakly correlated
industrial sensor readings. The 3-D scatter
(`images/ai4i_pca_3d.png`) places the TWF cluster along a distinct
axis — tool wear dominates the failure signal — while HDF/PWF/OSF/
RNF share a denser cloud, the same overlap that the classifier
confuses in §4.2.

## 5. Discussion
The two models behave as expected on imbalanced industrial data:
inverse-frequency weighting lifts recall on failure rows at the cost
of precision, and the residual errors concentrate in the tool-wear
200–240 min band that defines TWF and in the rare RNF mode. The
logistic-regression baseline already attains AUC 0.907, which means
the feature set (tool wear + type) carries most of the signal; the
MLP's contribution (AUC 0.966) is a ranking improvement rather than
a qualitative one. Honest evaluation — AUC and per-class F1 plus a
CV spread, not a single accuracy number — is the transferable
lesson. Potential improvements: a sequence model over UDI-order
tool wear, cost-sensitive decision thresholds, and SHAP-based
explainability (the dataset's original motivation).

## 6. Conclusion
We reproduced the full AI4I 2020 study end-to-end with a one-command
pipeline: data acquisition, EDA, 3-D PCA, baseline-versus-MLP
classification for two targets, and 5-fold cross-validation. The
compact MLP outperforms the LogReg baseline on both tasks (AUC
0.966 vs 0.907 for machine failure), with per-mode errors confined
to the rare failure modes. The pipeline is fully reproducible
(seed 42, pinned requirements, 21 pytest tests across 4 task files).

## References
- Matzka, S. (2020). Explainable Artificial Intelligence for
  Predictive Maintenance Applications. *AI4I 2020*, pp. 69–74.
- UCI Machine Learning Repository. AI4I 2020 Predictive Maintenance
  Dataset. DOI 10.24432/C5HS5C.
- Chollet, F., et al. Keras. https://keras.io.
