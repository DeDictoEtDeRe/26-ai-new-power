# Issue: Iris Dataset Neural Network Classification Analysis

## Overview
This issue involves a comprehensive analysis of the Iris dataset using neural networks for classification, including data exploration, visualization, model training, evaluation, and documentation.

## Tasks

### 1. Dataset Acquisition and Storage
- [x] **Download the Iris Dataset** from scikit-learn or UCI Machine Learning Repository
- [x] **Save the dataset** to the project directory in a structured format (e.g., `data/iris.csv`)
- [x] Ensure the data file is properly formatted and commit to version control

### 2. Data Exploration and 3D Visualization
- [x] **Describe the dataset fields, features, and classifications**
  - Features: sepal length, sepal width, petal length, petal width (all in cm)
  - Classes: Iris-setosa, Iris-versicolor, Iris-virginica
- [x] **Perform basic data exploration and inspection**
  - Check data shape, missing values, statistics
  - Display first few rows, class distribution
- [x] **Create 3D feature distribution plot using PCA**
  - Reduce 4 features to 3 principal components
  - Plot the data points colored by species/class
  - Save visualization as `images/iris_pca_3d.png`

### 3. Neural Network Classification Model
- [x] **Split data into training and test sets** (e.g., 80/20 or 70/30 split)
- [x] **Build a Neural Network model** using a deep learning framework (TensorFlow/Keras or PyTorch)
  - Input layer: 4 features
  - Hidden layers: configurable (e.g., 2 layers with ReLU activation)
  - Output layer: 3 units with softmax activation
- [x] **Train the model** on training data
- [x] **Evaluate the model** on test data
  - Calculate accuracy, precision, recall, F1-score
  - Generate confusion matrix
- [x] **Visualize important training and evaluation results**
  - Training vs validation loss/accuracy curves
  - Classification report visualization
  - Save plots as `images/iris_training_history.png` and `images/iris_confusion_matrix.png`

### 4. Generate "Mini Paper" Based on Experimental Results
- [x] **Write a summary paper** (800-1200 words) including:
  - Abstract
  - Introduction to Iris dataset
  - Methodology (data preprocessing, NN architecture, training parameters)
  - Results (accuracy, key observations from visualizations)
  - Discussion (model performance, potential improvements)
  - Conclusion
- [x] **Save the paper** as `docs/iris_nn_analysis.md` or `.pdf`

## Success Criteria
✅ Iris dataset saved to project directory (data/iris.csv)
✅ Data exploration and 3D PCA visualization completed (images/iris_pca_3d.png)
✅ Neural network model trained and evaluated (96.67% test accuracy)
✅ All visualizations saved and clear (3 plot files in images/)
✅ Mini paper generated with experimental findings (docs/iris_nn_analysis.md)
✅ TDD test suite implemented (15 tests across 4 task files)

## Status: ✅ Complete
All tasks finished and committed (see `docs/iris_nn_analysis.md` for results: test accuracy 96.67%, macro F1 0.9666, 5-fold CV 96.00% ± 5.33%). All 15 pytest tests pass.

## TDD Test Results
All 15 TDD tests pass, confirming the implementation meets requirements:
- **Task 1 (5 tests)**: Dataset existence, columns, shape, class distribution, data types
- **Task 2 (3 tests)**: PCA reduction to 3 components, 3D plot existence, basic exploration
- **Task 3 (3 tests)**: Model architecture, training capability, evaluation metrics
- **Task 4 (4 tests)**: Paper existence, section structure, key results, minimum length

## Technical Notes
- TDD methodology used: Red-Green-Refactor cycle
- Test framework: pytest with TensorFlow/Keras backend
- All random seeds set (42) for reproducibility
- Hyperparameters documented: 80/20 train-test split, 100 epochs, batch size 16, Adam optimizer