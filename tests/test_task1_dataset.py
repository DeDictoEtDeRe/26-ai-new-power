"""Test file for Task 1: Iris Dataset Acquisition and Storage"""
import os
import pytest
import pandas as pd
import numpy as np

def test_iris_csv_exists():
    """Test that iris.csv exists in the data directory."""
    assert os.path.exists('data/iris.csv'), "data/iris.csv should exist"

def test_iris_csv_columns():
    """Test that iris.csv has the correct columns."""
    df = pd.read_csv('data/iris.csv')
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 
                        'petal length (cm)', 'petal width (cm)', 'species']
    assert list(df.columns) == expected_columns, f"Expected {expected_columns}, got {list(df.columns)}"

def test_iris_csv_shape():
    """Test that iris.csv has 150 rows."""
    df = pd.read_csv('data/iris.csv')
    assert df.shape[0] == 150, f"Expected 150 rows, got {df.shape[0]}"
    assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"

def test_iris_csv_class_distribution():
    """Test that iris.csv has balanced class distribution."""
    df = pd.read_csv('data/iris.csv')
    expected_classes = ['setosa', 'versicolor', 'virginica']
    for cls in expected_classes:
        count = len(df[df['species'] == cls])
        assert count == 50, f"Expected 50 {cls} samples, got {count}"

def test_iris_csv_data_types():
    """Test that iris.csv has correct data types."""
    df = pd.read_csv('data/iris.csv')
    for col in ['sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'petal width (cm)']:
        assert df[col].dtype in [np.float64, float], f"{col} should be float, got {df[col].dtype}"