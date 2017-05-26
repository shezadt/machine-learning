#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Part I : Algorithm

class LogisticRegression():

    def __init__(self, alpha=0.1, nb_iter=1000):

        # Define the learning rate
        self.alpha = alpha

        # Define the number of iterations
        self.nb_iter = nb_iter

        # Define the weights of the logistic regression
        self.weights = None

    # Define the sigmoid function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def fit(self, X_train, y_train):

        # Add a column of ones a first column for bias weights
        X_train_features = np.insert(X_train, 0, 1, axis=1)

        # Define the weight vector with the bias
        _, n_features = X_train_features.shape
        weights = np.zeros(n_features)

        # Loop on the number of iterations
        for i in xrange(self.nb_iter):

            # Make a prediction with the current weights
            predictions = self._sigmoid(np.dot(X_train_features, weights))

            # Update the weights with the gradient
            target_error = predictions - y_train
            gradient = np.dot(X_train_features.T, target_error)
            weights += self.alpha * gradient

        # Save the weights
        self.weights = weights

    def predict(self, X_test):

        # Get the estimation of the probability
        estimation = np.dot(np.insert(X_test, 0, 1, axis=1), self.weights)
        y_pred = np.round(self._sigmoid(estimation)).astype(int)

        return y_pred

# Part II : An example

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

# Scale the data
X_scaled = preprocessing.scale(X)

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Machine learning algorithm
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)
