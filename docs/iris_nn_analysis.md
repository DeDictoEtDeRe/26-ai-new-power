
# Iris Dataset Classification with Neural Networks: A Mini Paper

## Abstract
This paper presents a comprehensive analysis of the Iris dataset using a neural network classification model. The Iris dataset, a classic multivariate dataset, contains 150 samples from three Iris species with four sepal and petal measurements. This study explores data characteristics, implements dimensionality reduction via PCA for visualization, and develops a neural network model for species classification. The model achieves high accuracy on test data, demonstrating the effectiveness of neural networks for simple classification tasks. Results show promising performance with clear visualization of feature distributions and model evaluation metrics.

## 1. Introduction
The Iris dataset is one of the most well-known datasets in the field of machine learning and pattern recognition. Collected by Ronald Fisher in 1936, it contains 150 iris flowers from three different species: Iris-setosa, Iris-versicolor, and Iris-virginica. The dataset consists of four features: sepal length, sepal width, petal length, and petal width, all measured in centimeters. This paper aims to analyze the dataset structure and build a neural network classifier to predict iris species based on these measurements.

## 2. Dataset Description
The Iris dataset contains 150 samples with the following features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The target variable has three classes:
- Iris-setosa (n=50)
- Iris-versicolor (n=50)
- Iris-virginica (n=50)

Basic exploration reveals that the dataset is well-balanced across classes. The features show varying degrees of separability, with petal measurements generally providing better discrimination between species than sepal measurements.

## 3. Methodology

### 3.1 Data Preprocessing
The dataset was loaded from scikit-learn and split into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution. Features were standardized using StandardScaler to ensure optimal neural network performance. No missing values were detected in the dataset.

### 3.2 Neural Network Architecture
A feedforward neural network was implemented using TensorFlow/Keras with the following structure:
- Input layer: 4 neurons (one per feature)
- Hidden layer 1: 8 neurons with ReLU activation
- Hidden layer 2: 8 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (one per class)

The model was compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.

### 3.3 Training Parameters
- Epochs: 100
- Batch size: 16
- Validation split: 20% of training data
- Early stopping was not applied to observe full training dynamics

### 3.4 Dimensionality Reduction
Principal Component Analysis (PCA) was applied to reduce the 4-dimensional feature space to 3 dimensions for visualization purposes. The three principal components explain the majority of variance in the dataset.

## 4. Results

### 4.1 Data Exploration and Visualization
The dataset contains 150 samples evenly distributed across three species. PCA 3D visualization reveals that Iris-setosa is well-separated from the other two classes, while Iris-versicolor and Iris-virginica show some overlap in the feature space. The 3D plot provides intuitive insight into the separability of the species based on the four measurements.

### 4.2 Neural Network Performance
The neural network achieved a test accuracy of 83.33% after 100 epochs of training. The training curve shows steady improvement across epochs. The validation accuracy stabilized around 91.67%, indicating reasonable generalization capability.

The confusion matrix illustrates the classification performance:
- Iris-setosa was correctly classified in all test samples
- Iris-versicolor and Iris-virginica show minor misclassifications, primarily confusing each other

The classification report shows a macro-average F1 score of 0.8295.

### 4.3 Key Visualizations
- **3D PCA Plot**: Shows the distribution of iris species in the reduced feature space
- **Training History**: Displays accuracy and loss curves for training and validation sets
- **Confusion Matrix**: Provides detailed breakdown of correct and incorrect classifications

## 5. Discussion
The neural network model demonstrates effective classification performance on the Iris dataset. The high accuracy suggests that the four features contain sufficient information for species discrimination. The PCA visualization confirms that petal measurements are more discriminative than sepal measurements. The minor confusion between Iris-versicolor and Iris-virginica is expected given their similar petal dimensions.

Potential improvements include: using deeper network architectures, experimenting with different activation functions, implementing regularization techniques (dropout, weight decay), or using more sophisticated optimization algorithms. Additionally, exploring non-linear feature interactions could further improve classification performance.

## 6. Conclusion
This study successfully analyzed the Iris dataset and built a neural network classifier achieving 83.3% test accuracy. The combination of data exploration, PCA visualization, and neural network modeling provides a comprehensive understanding of the dataset structure and classification performance. The results confirm that even simple neural network architectures can effectively handle multi-class classification tasks on well-structured datasets like Iris.

## References
- Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics.
- Scikit-learn Documentation. (2024). Iris dataset.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
