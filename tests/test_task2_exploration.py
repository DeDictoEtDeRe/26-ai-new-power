"""Test file for Task 2: Data Exploration and 3D PCA Visualization"""
import os
import pytest
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

def test_pca_reduction():
    """Test that PCA can reduce Iris data to 3 components."""
    iris = datasets.load_iris()
    X = iris.data
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    assert X_pca.shape == (150, 3), f"Expected shape (150, 3), got {X_pca.shape}"
    # Check explained variance
    total_variance = sum(pca.explained_variance_ratio_)
    assert total_variance > 0.95, f"Expected >95% variance, got {total_variance:.2%}"

def test_pca_3d_plot_exists():
    """Test that the 3D PCA plot file exists."""
    assert os.path.exists('images/iris_pca_3d.png'), "images/iris_pca_3d.png should exist"
    # Check file size is reasonable (not empty)
    file_size = os.path.getsize('images/iris_pca_3d.png')
    assert file_size > 1000, f"Plot file too small: {file_size} bytes"

def test_data_exploration_basic():
    """Test basic data exploration functions."""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    assert X.shape == (150, 4), f"Expected (150, 4), got {X.shape}"
    assert len(np.unique(y)) == 3, f"Expected 3 classes, got {len(np.unique(y))}"