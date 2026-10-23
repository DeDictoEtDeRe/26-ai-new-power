
# Iris Species Classification with a Feed-Forward Neural Network: A Short Paper

## Abstract
We present a complete small-scale machine-learning study on the classic Iris
dataset: 150 flower measurements across three species. After basic exploratory
data analysis, we reduce the four feature dimensions to three principal
components (99.48% of total variance retained) for an intuitive 3D
visualization, and train a compact feed-forward neural network
(4-16-8-3, ReLU hidden units, softmax output) with a stratified 80/20
train-test split, standardization, and early stopping. The network reaches
96.67% test accuracy (macro-F1 = 0.9666) and a 5-fold cross-validation
estimate of 96.00% ± 5.33%. The two error cases involve confusion between
Iris versicolor and Iris virginica, consistent with the overlap visible in
the PCA projection. The results confirm that even a 243-parameter model
classifies Iris reliably, while honest evaluation (held-out test set plus
cross-validation) matters more than a single optimistic accuracy figure.

## 1. Introduction
The Iris dataset, collected by Ronald Fisher (1936), is one of the most
familiar benchmarks in machine learning and pattern recognition. It contains
150 observations from three Iris species — Iris setosa, Iris versicolor, and Iris virginica
— each described by four measurements: sepal length, sepal width, petal
length, and petal width, all in centimetres. Although tiny by modern
standards, the dataset is an excellent vehicle for a full
data-to-deployment workflow: cleaning and exploration, dimensionality
reduction for visualization, and supervised classification. This paper
re-runs that workflow end-to-end in this repository (the pipeline is
reproducible with a single command), evaluates a neural-network classifier,
and reports the findings.

## 2. Dataset Description
`data/iris.csv` holds 150 rows and 5 columns:

| Field | Type | Unit | Description |
| --- | --- | --- | --- |
| `sepal length (cm)` | float | cm | Length of the sepal |
| `sepal width (cm)` | float | cm | Width of the sepal |
| `petal length (cm)` | float | cm | Length of the petal |
| `petal width (cm)` | float | cm | Width of the petal |
| `species` | string | — | `Iris setosa`, `Iris versicolor`, or `Iris virginica` |

The classes are perfectly balanced: 50 samples per species (33.3% each). The
data inspection summary (`results/eda_summary.txt`) reports no missing
values and one fully duplicated row (a known characteristic of the original
release). Descriptive statistics show that petal measurements span much
wider ranges (petal length 1.0–6.9 cm) than sepal measurements
(4.3–7.9 cm), and the correlation matrix reveals strong positive
correlations between petal length and width (0.96) and between sepal and
petal length (0.87). Mean measurements separate the species clearly: Iris
setosa has the smallest petals (mean petal length ≈ 1.46 cm), Iris versicolor
is intermediate (≈ 4.26 cm), and Iris virginica is the largest (≈ 5.14 cm),
with versicolor and virginica overlapping noticeably.

## 3. Methodology

### 3.1 Data Inspection and EDA
`src/eda.py` produces a plain-text summary (shape, dtypes, head, missing and
duplicate counts, class distribution, descriptive statistics, Pearson
correlations, and per-species means) and three figures: a 4×4 pair plot
(`images/iris_pairplot.png`), a correlation heatmap
(`images/iris_correlation.png`), and box-and-strip plots of each feature by
species (`images/iris_feature_distributions.png`). These confirm that
petal dimensions are the strongest discriminative features.

### 3.2 Dimensionality Reduction (PCA)
Principal Component Analysis reduces the 4-D feature space to 3 principal
components for visualization. The explained-variance ratios are
0.730, 0.229, and 0.037 — cumulatively 73.0%, 95.8%, and 99.5% of the
total variance. The resulting 3-D scatter plot
(`images/iris_pca_3d.png`, saved with loadings in `results/pca.json`) shows
Iris setosa forming a compact, isolated cluster, while versicolor and virginica
form two elongated, partially overlapping clouds — matching the
interpretation of the raw feature plots.

### 3.3 Data Splitting and Preprocessing
The dataset is split 80/20 with stratification (seed 42): 120 training
rows (40 per class) and 30 test rows (10 per class). Features are
standardized with a `StandardScaler` fitted on the training split only,
preventing any leakage of test-set statistics. An additional 20% of the
training rows (24 samples) are held out as a validation set to drive
early stopping.

### 3.4 Neural-Network Architecture
The classifier is a small feed-forward network built with Keras
(`models/iris_nn.keras`): an input layer of 4 units, two hidden layers of
16 and 8 ReLU units, and a 3-way softmax output — 243 trainable
parameters in total. It is compiled with the Adam optimizer
(learning rate 1e-3) and sparse categorical cross-entropy.

### 3.5 Training Protocol
Training runs for up to 300 epochs with batch size 16, with two
callbacks: early stopping on validation loss (patience 40, best weights
restored) and a learning-rate reduction on plateau (factor 0.5, patience
15, minimum 1e-4). All seeds are fixed (42) for reproducibility. In this
run the model trained the full 300 epochs, with the best-epoch criterion
falling on epoch 300.

### 3.6 Evaluation
Generalization is assessed two ways: (i) the fixed 80/20 test set,
reporting accuracy, per-class precision/recall/F1, a confusion matrix, and
log loss; and (ii) a stratified 5-fold cross-validation over the full
dataset, which trains five independent models to estimate how much
performance depends on the split.

## 4. Results

### 4.1 EDA and PCA Observations
The 3-D PCA plot is the clearest single picture of class geometry: Iris
setosa is trivially separable, whereas versicolor/virginica overlap — the
confusion pattern later seen in the confusion matrix. Feature-distribution
figures (box plots) show that petal width alone nearly separates Iris
setosa from the other two species, and separates virginica from versicolor
only imperfectly.

### 4.2 Neural-Network Performance
On the held-out test set (30 samples):

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9667 (29/30) |
| Macro precision | 0.9697 |
| Macro recall | 0.9667 |
| Macro F1 | 0.9666 |
| Test log loss | 0.0799 |

Per class (from `results/classification_report.txt`): Iris setosa was
classified perfectly (precision = recall = 1.0000). Iris versicolor had one
miss (recall 0.9000, F1 0.9474) — one flower predicted as Iris virginica —
and Iris virginica had one miss on the precision side (0.9091, F1 0.9524)
because that versicolor flower was mislabeled as virginica. The two errors
are mutually consistent: the network confused one versicolor with one
virginica, which is exactly where the PCA projection shows overlap.
Validation accuracy at the best epoch was 0.9583.

The 5-fold cross-validation produced fold accuracies of
1.0000, 1.0000, 0.8667, 1.0000, 0.9333 — mean 0.9600 with a standard
deviation of 0.0533. With only 30 test samples per split, one misclassified
flower changes a fold's accuracy by 3.3 percentage points, so the spread
is expected and worth reporting rather than hiding.

### 4.3 Key Visualizations
- `images/iris_pca_3d.png` — 3-D PCA scatter, coloured by species
- `images/iris_training_history.png` — accuracy and loss curves with best-epoch marker
- `images/iris_confusion_matrix.png` — annotated test-set confusion matrix
- `images/iris_classification_report.png` — precision/recall/F1 table

## 5. Discussion
A 243-parameter network comfortably solves this problem: 96.67% on a held-out
test set and 96.00% ± 5.33% across five folds. Two points deserve attention.
First, evaluation honesty: with 30 test samples, single-split accuracy is
noisy, and the cross-validation spread (±5.3 points) quantifies that noise;
any headline number should be quoted with it. Second, error structure: all
confusion is Iris versicolor↔Iris virginica, matching the PCA overlap and the weak
separation of the two species in sepal width. Setosa is effectively
trivial for any reasonable model. Potential improvements include adding
regularization (dropout, weight decay), trying deeper or wider hidden layers,
calibrating probabilities, and reporting confidence intervals over many
seeds; none is likely to move the ceiling much, since petal features alone
already nearly separate the classes.

## 6. Conclusion
We reproduced the full Iris study end-to-end: dataset acquisition, EDA,
3-D PCA visualization, neural-network training with an 80/20 stratified
split, and multi-faceted evaluation. The compact MLP reached 96.67% test
accuracy and 96.00% ± 5.33% cross-validation accuracy, with all residual
errors confined to the Iris versicolor/Iris virginica boundary. Even a minimal
neural network classifies Iris reliably; the more instructive lessons are
the importance of honest evaluation (cross-validation, per-class
breakdowns) and of tying model errors back to the geometry of the data.

## References
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic
  problems. *Annals of Eugenics*, 7(2), 179–188.
- scikit-learn documentation. Iris dataset (UCI ML Repository #53).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
  MIT Press.
- Abadi, M., et al. (2015). TensorFlow: Large-scale machine learning.
  *arXiv:1610.06516* (framework lineage of the Keras API used here).
