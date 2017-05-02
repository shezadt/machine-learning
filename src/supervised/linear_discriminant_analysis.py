#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Part I : Problematic

# Load the libraries
from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal as multi_normal

# Load the iris data set
iris = datasets.load_iris()

# Define the features
X = iris.data
y = iris.target

# Name of the labels and their number
name_labels = np.unique(y)
n_labels = len(name_labels)

# Number of features
n_features = X.shape[1]

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Number of train sample
n_train = len(X_train)

# Number of test samples
n_test = len(X_test)

# Part II : Algorithm

# Compute the prior
prior = np.bincount(y_train) / n_train

# Compute the parameters of the likelihood
mean_labels = np.zeros((n_labels, n_features))
cov_labels = np.zeros((n_labels, n_features, n_features))

for i, label in enumerate(name_labels):

    # Compute the mean of the train samples by label
    mean_labels[i, :] = X_train[np.where(y_train == label)].mean()

    # Compute the covariance matrix of the train samples by label
    X_train_label_scaled = scale(X_train[np.where(y_train == label)])
    cov_labels[i] = np.cov(X_train_label_scaled.T)

# Compute the global covariance matrix
cov = np.zeros((n_features, n_features))

for i in range(n_labels):

    cov += prior[i] * cov_labels[i]

# Define the prediction vector
y_pred = np.zeros(n_test)

# Predict the label for each test sample
for i, test_sample in enumerate(X_test):

    # Define the numerator vector of the Bayes theorem
    numerator = np.zeros(n_labels)

    # Compute the numerator for each label
    for j in range(n_labels):

        # Compute the likelihood
        likelihood = multi_normal(mean_labels[j], cov).pdf(test_sample)

        # Compute the numerator
        numerator[j] = prior[j] * likelihood

    # Compute the evidence of the Bayes theorem
    evidence = numerator.sum()

    # Compute the posterior using Bayes theorem
    posterior = numerator / evidence

    # Chose the label which maximizes the posterior probability
    y_pred[i] = np.argmax(posterior)

# Compute the accuracy score
accuracy = accuracy_score(y_test, y_pred)
