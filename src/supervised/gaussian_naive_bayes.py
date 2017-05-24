#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats

# Part I : Algorithm

class GaussianNB():

    def __init__(self):

        # Name of the labels
        self.name_labels = None

        # Number of labels
        self.n_labels = None

        # Mean of all the features
        self.features_mean = None

        # Standard deviations of all the features
        self.features_std = None

        # Prior
        self.prior = None

    def _compute_likelihood(self, means, stds, obs):

        probability = 1

        # The naive assumption : each feature in uncorrelated from each other
        for z in range(len(obs)):
            probability *= scipy.stats.norm(means[z], stds[z]).pdf(obs[z])

        return probability

    def fit(self, X_train, y_train):

        # Name of the labels and their number
        self.name_labels = np.unique(y_train)
        self.n_labels = len(self.name_labels)

        # Dimension of the space
        n_train, n_features = X_train.shape

        # Compute the parameters of the likelihood from the train set
        features_mean = np.zeros((self.n_labels, n_features))
        features_std = np.zeros((self.n_labels, n_features))

        for i, label in enumerate(self.name_labels):

            # Compute the mean of each feature for each label
            features_mean[i, :] = X_train[np.where(y_train == label)].mean()

            # Compute the variance of each feature for each label
            features_std[i, :] = X_train[np.where(y_train == label)].std()

        # Save all the parameters
        self.features_mean = features_mean
        self.features_std = features_std
        self.prior = np.bincount(y_train) / n_train

    def predict(self, X_test):

        # Number of test samples
        n_test = len(X_test)

        # Define the prediction vector
        y_pred = np.zeros(n_test)

        # Predict the label for each test sample
        for i, test_sample in enumerate(X_test):

            # Define the numerator vector of the Bayes theorem
            numerator = np.zeros(self.n_labels)

            # Compute the numerator for each label
            for j in range(self.n_labels):

                # Compute the likelihood
                likelihood = self._compute_likelihood(self.features_mean[j, :],
                                                self.features_std[j, :],
                                                test_sample)

                # Compute the numerator
                numerator[j] = self.prior[j] * likelihood

            # Compute the evidence of the Bayes theorem
            evidence = numerator.sum()

            # Compute the posterior using Bayes theorem
            posterior = numerator / evidence

            # Chose the label which maximizes the posterior probability
            y_pred[i] = np.argmax(posterior)

        return y_pred

# Part II : An example

# Load the iris data set
iris = datasets.load_iris()

# Define the features and the target
X = iris.data
y = iris.target

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Machine learning algorithm
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)
