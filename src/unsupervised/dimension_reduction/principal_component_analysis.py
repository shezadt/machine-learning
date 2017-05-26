#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Part I : Algorithm

class PCA():

    def __init__(self, n_components=2):

        # Define the number of components to keep
        self.n_components = n_components

        # Define the eigen vectors base
        self.eigen_vectors_base = None

    def fit(self, X_train):

        # Compute the covariance matrix
        cov_X = np.cov(np.transpose(X_train))

        # Compute the eigen values & the eigen vectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eig(cov_X)

        # Sort the eigen values by decreasing order
        ev_indices = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[ev_indices]

        # Select the first NB_COMPONENTS largest eigen values
        eigen_values_selected = eigen_values[:self.n_components]

        # Select the eigen vectors corresponding to the eigen values selected
        eigen_vectors_base = eigen_vectors[:, ev_indices][:, :self.n_components]

        # Save the base
        self.eigen_vectors_base = eigen_vectors_base

    def transform(self, X):

        # Project the data onto principal components
        X_transformed = X.dot(self.eigen_vectors_base)

        return X_transformed

# Part II : An example

# Load the MNIST data set
digits = datasets.load_digits()

# Define the features and the target
X = digits.data
y = digits.target

# Center the data but do not scale this particular data set because
# all the features are measured in the same unit
X_centered = preprocessing.scale(X, with_std=False)

# Machine learning algorithm
pca = PCA(n_components=2)
pca.fit(X_centered)
X_transformed = pca.transform(X)

# Visualize the first two components
fig, ax = plt.subplots()

ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)

plt.show()
