#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Part I : Problematic

# Load the libraries
import numpy as np

from sklearn import datasets
from sklearn import preprocessing

import matplotlib.pyplot as plt

# Load the MNIST data set
digits = datasets.load_digits()

# Define the features and the target
X = digits.data
y = digits.target

# Part II : Algorithm

# Define the number of components to keep
NB_COMPONENTS = 5

# Center the data but do not scale this particular data set because
# all the features are measured in the same unit
X_centered = preprocessing.scale(X, with_std=False)

# Compute the covariance matrix
cov_X = np.cov(np.transpose(X_centered))

# Compute the eigen values and the eigen vectors of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(cov_X)

# Sort the eigen values by decreasing order
ev_indices = eigen_values.argsort()[::-1]
eigen_values = eigen_values[ev_indices]

# Select the first NB_COMPONENTS largest eigen values
eigen_values_selected = [:NB_COMPONENTS]

# Select the eigen vectors corresponding to the eigen values selected
eigen_vectors_selected = eigen_vectors[:, ev_indices][:, :NB_COMPONENTS]

# Project the data onto principal components
X_transformed = X.dot(eigen_vectors_selected)

# Visualize the first two components
fig, ax = plt.subplots()

ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)

plt.show()
