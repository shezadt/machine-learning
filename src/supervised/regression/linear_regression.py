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

class LinearRegression():

    def __init__(self):

        # The weights of the linear regression
        self.weights = None

    def fit(self, X_train, y_train):

        # Add a column of ones a first column for bias weights
        X_train_features = np.insert(X_train, 0, 1, axis=1)

        # Compute the weights from the train set
        cov_X = X_train_features.T.dot(X_train_features)
        self.weights = np.linalg.inv(cov_X).dot(X_train_features.T).dot(y_train)

    def predict(self, X_test):

        # Compute the dot product
        y_pred = np.insert(X_test, 0, 1, axis=1).dot(self.weights)

        return y_pred

# Part II : An example

# Load the Boston data set
boston = datasets.load_boston()

# Define the features
X = boston.data
y = boston.target

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Machine learning algorithm
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Compute the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Compute the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Visualize the results
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Linear regression on the Boston data set')
plt.xlabel('Real values')
plt.ylabel('Prediction')
plt.show()
