#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Part I : Problematic

# Load the libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set the seed
np.random.seed(1)

# Load the iris data set
iris = datasets.load_iris()

# Define the features
X = iris.data

# Get the number of observations
N_obs = len(X)

# Shuffle the data
X = X[np.random.permutation(N_obs)]

# Part II : Algorithm

# Define the number of clusters
K = 3

# Define the number of iterations
N_iter = 100

# Initialize the centroids as random samples from the data
centroids = X[np.random.randint(N_obs, size=K)]

# Define the clusters vector
clusters = np.zeros(N_obs)

# Loop on the number of iterations
for i in range(N_iter):

    # For each sample of the data
    for j, sample in enumerate(X):

        # Compute the distance between the sample and the centroids
        distances = np.linalg.norm(centroids - sample, axis=1)

        # Assign the sample with the closest centroid
        clusters[j] = np.argmin(distances)

    # Update the centroids
    for z in range(K):
        centroids[z] = X[np.where(clusters == z)].mean(axis=0).round(1)

# Visualize the clusters
fig, ax = plt.subplots()

ax.scatter(X[:, 2], X[:, 3], c=clusters)

plt.show()
