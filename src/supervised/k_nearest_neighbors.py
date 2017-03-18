#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris data set
iris = datasets.load_iris()

# Define the features and the target
X = iris.data
y = iris.target

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=0)
# Number of training samples
N_train = len(X_train)

# Number of test samples
N_test = len(X_test)

# Define the number of neighbors
K = 3

# Define the prediction vector
y_pred = np.zeros(N_test)

# Predict the label for each test sample
for i, test_sample in enumerate(X_test):
    
    # Define the distance vector with the train set
    distances = np.zeros(N_train)
    
    # Compute the distance with each train sample
    for j, train_sample in enumerate(X_train):
        distances[j] = np.linalg.norm(test_sample - train_sample)
    
    # Sort the distances to get the nearest neighbors
    nn_indices = distances.argsort()
    nn_label = y_train[nn_indices]
    
    # Select the K first nearest neighbors
    nn_y = nn_label[:K]
    
    # Do a majority vote to classify
    y_pred[i] = np.argmax(np.bincount(nn_y))

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)
