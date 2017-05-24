#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Part I : Algorithm

class NearestNeighbors():

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def predict(self, X_train, X_test, y_train):

        # Define the prediction vector
        y_pred = np.zeros(len(X_test))

        # Predict the label for each test sample
        for i, test_sample in enumerate(X_test):

            # Compute the distance with each train sample
            distances = np.linalg.norm(X_train - test_sample, axis=1)

            # Sort the distances to get the nearest neighbors
            nn_indices = distances.argsort()
            nn_label = y_train[nn_indices]

            # Select the n_neighbors first nearest neighbors
            nn_y = nn_label[:self.n_neighbors]

            # Do a majority vote to classify
            y_pred[i] = np.argmax(np.bincount(nn_y))

        return y_pred

# Part II : An example

# Load the iris data set
iris = datasets.load_iris()

# Define the features and the target
X = iris.data
y = iris.target

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Machine learning algorithm
clf = NearestNeighbors(n_neighbors=3)
y_pred = clf.predict(X_train, X_test, y_train)

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)
