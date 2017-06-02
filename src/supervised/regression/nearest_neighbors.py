#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Part I : Algorithm

class NearestNeighbors():

    def __init__(self, n_neighbors=1):

        # Number of neighbors
        self.n_neighbors = n_neighbors

    def fit_predict(self, X_train, y_train, X_test):

        # Define the prediction vector
        y_pred = np.zeros(len(X_test))

        # Predict the label for each test sample
        for i, test_sample in enumerate(X_test):

            # Compute the distance with each train sample
            distances = np.linalg.norm(X_train - test_sample, axis=1)

            # Sort the distances to get the nearest neighbors
            nn_indices = distances.argsort()
            nn_label = y_train[nn_indices]

            # Select the first n_neighbors nearest neighbors
            nn_y = nn_label[:self.n_neighbors]

            # Compute the mean of the selected nearest neighbors
            y_pred[i] = nn_y.mean()

        return y_pred

# Part II : An example

# Load the Boston data set
boston = datasets.load_boston()

# Define the features and the target
X = boston.data
y = boston.target

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Machine learning algorithm
n_neighbors = 3
reg = NearestNeighbors(n_neighbors=n_neighbors)
y_pred = reg.fit_predict(X_train, y_train, X_test)

# Compute the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Compute the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Visualize the results
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Nearest neighbors with K = %d' % n_neighbors)
plt.xlabel('Real values')
plt.ylabel('Prediction')
plt.show()
