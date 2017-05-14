#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Part I : Problematic

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the MNIST data set
digits = datasets.load_digits()

# Define the features and the target
digit1 = 5
digit2 = 6

X = digits.data
y = digits.target

X = X[(y == digit1) | (y == digit2)]
y = y[(y == digit1) | (y == digit2)]

y[y == digit1] = 0
y[y == digit2] = 1

# Number of features
n_features = X.shape[1]

# Scale the data
X_scaled = preprocessing.scale(X)

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Part II : Algorithm

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(x))

# Define the learning rate
ALPHA = 0.1

# Define the number of iterations
NB_ITER = 1000

# Add a column of ones a first column for bias weights
X_train_features = np.insert(X_train, 0, 1, axis=1)

# Define the weight vector (+ 1 for the bias weights)
weights = np.zeros(n_features + 1)

# Loop on the number of iterations
for i in xrange(NB_ITER):

    # Make a prediction with the current weights
    predictions = sigmoid(np.dot(X_train_features, weights))

    # Update the weights with the gradient
    target_error = predictions - y_train
    gradient = np.dot(X_train_features.T, target_error)
    weights += ALPHA * gradient

# Make the prediction on the test set
estimations = np.dot(np.insert(X_test, 0, 1, axis=1), weights)
y_pred = np.round(sigmoid(estimations)).astype(int)

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)
