"""Test file for Task 3: Neural Network Classification Model"""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers

def test_nn_model_architecture():
    """Test that the neural network model has the expected architecture."""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(4,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Check model has correct layers
    assert len(model.layers) == 3, f"Expected 3 layers, got {len(model.layers)}"
    assert model.layers[0].units == 8, f"Expected 8 units in layer 1, got {model.layers[0].units}"
    assert model.layers[1].units == 8, f"Expected 8 units in layer 2, got {model.layers[1].units}"
    assert model.layers[2].units == 3, f"Expected 3 units in output layer, got {model.layers[2].units}"

def test_model_can_train():
    """Test that the model can be trained without errors."""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_onehot = __import__('tensorflow').keras.utils.to_categorical(y_train, 3)
    
    model = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(4,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train_scaled, y_train_onehot,
                        epochs=5, batch_size=16,
                        validation_split=0.2,
                        verbose=0)
    
    # Check training completed
    assert len(history.history['accuracy']) > 0, "Training history should have accuracy records"
    assert history.history['accuracy'][-1] > 0, "Final training accuracy should be > 0"

def test_model_evaluation():
    """Test that the model can be evaluated on test data."""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    from tensorflow.keras import models, layers
    y_train_onehot = __import__('tensorflow').keras.utils.to_categorical(y_train, 3)
    
    model = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(4,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train_scaled, y_train_onehot, epochs=5, batch_size=16, verbose=0)
    
    loss, accuracy = model.evaluate(X_test_scaled, 
                                    __import__('tensorflow').keras.utils.to_categorical(y_test, 3),
                                    verbose=0)
    
    # Check evaluation results
    assert accuracy > 0.5, f"Expected accuracy > 0.5, got {accuracy}"
    assert loss < 1.0, f"Expected loss < 1.0, got {loss}"