#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal as mn

# Part I : Algorithm

class LinearDiscriminantAnalysis():

    def __init__(self):

        # Name of the labels
        self.name_labels = None

        # Number of labels
        self.n_labels = None

        # Mean by label
        self.mean_labels = None

        # Covariance matrix
        self.cov = None

        # Prior
        self.prior = None

    def fit(self, X_train, y_train):

        # Name of the labels and their number
        self.name_labels = np.unique(y_train)
        self.n_labels = len(self.name_labels)

        # Dimension of the space
        n_train, n_features = X_train.shape

        # Compute the prior
        self.prior = np.bincount(y_train)/ n_train

        # Compute the parameters of the likelihood
        mean_labels = np.zeros((self.n_labels, n_features))
        cov_labels = np.zeros((self.n_labels, n_features, n_features))

        for i, label in enumerate(self.name_labels):

            # Compute the mean of the train samples by label
            mean_labels[i, :] = X_train[np.where(y_train == label)].mean()

            # Compute the covariance matrix of the train samples by label
            X_train_label_scaled = scale(X_train[np.where(y_train == label)])
            cov_labels[i] = np.cov(X_train_label_scaled.T)

        # Save the mean by labels
        self.mean_labels = mean_labels

        # Compute the global covariance matrix
        cov = np.zeros((n_features, n_features))

        for i in range(self.n_labels):

            cov += self.prior[i] * cov_labels[i]

        # Save the covariance matrix
        self.cov = cov

    def predict(self, X_test):

        # Define the prediction vector
        y_pred = np.zeros(len(X_test))

        # Predict the label for each test sample
        for i, test_sample in enumerate(X_test):

            # Define the numerator vector of the Bayes theorem
            numerator = np.zeros(self.n_labels)

            # Compute the numerator for each label
            for j in range(self.n_labels):

                # Compute the likelihood
                likelihood = mn(self.mean_labels[j], self.cov).pdf(test_sample)

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
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)
