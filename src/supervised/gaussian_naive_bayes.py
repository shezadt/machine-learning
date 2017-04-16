#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Part I : Problematic

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris data set
iris = datasets.load_iris()

# Define the features
X = iris.data
y = iris.target

# Name of the labels
name_labels = np.unique(y)
n_labels = len(name_labels)

# Number of features
n_features = X.shape[1]

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Number of test samples
n_test = len(X_test)

# Part II : Algorithm

# Define the gaussian probability distribution
def gaussian_density(mean, var, x):

    coeff = (1 / (np.sqrt((2.0 * np.pi) * var)))
    exponent = np.exp(-((x - mean)**2 / (2 * var)))

    return coeff * exponent

# Define the likelihood function
def compute_likelihood(means, vars, obs):

    probability = 1

    # The naive assumption : each feature in uncorrelated from each other
    for z in range(obs.shape[0]):
        probability *= gaussian_density(means[z], vars[z], obs[z])

    return probability

# Compute the parameters of the likelihood from the train set
features_mean = np.zeros((n_labels, n_features))
features_var = np.zeros((n_labels, n_features))

for i, label in enumerate(name_labels):

    # Compute the mean of each feature for each label
    features_mean[i, :] = X_train[np.where(y_train == label)].mean()

    # Compute the variance of each feature for each label
    features_var[i, :] = X_train[np.where(y_train == label)].var()

# Compute the prior
prior = np.bincount(y_train)

# Define the prediction vector
y_pred = np.zeros(n_test)

# Predict the label for each test sample
for i, test_sample in enumerate(X_test):

    # Define the posterior vector
    posterior = np.zeros(n_labels)

    # Compute the posterior for each label
    for j in range(n_labels):

        # Compute the likelihood
        likelihood = compute_likelihood(features_mean[j, :],
                                        features_var[j, :],
                                        test_sample)

        # Compute the posterior using Bayes theorem
        posterior[j] = prior[j] * likelihood

    y_pred[i] = np.argmax(posterior)

accuracy = accuracy_score(y_test, y_pred)
