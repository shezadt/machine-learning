#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Part I : Algorithm

class KMeans():

    def __init__(self, n_clusters=8, max_iter=300):

        # The number of clusters
        self.n_clusters = n_clusters

        # The number of iterations
        self.max_iter = max_iter

    def fit_predict(self, X_train):

        # Get the number of observations
        N_obs = len(X_train)

        # Initialize the centroids as random samples from the data
        centroids = X_train[np.random.randint(N_obs, size=self.n_clusters)]

        # Define the clusters vector
        clusters = np.zeros(N_obs)

        # Loop on the number of iterations
        for i in range(self.max_iter):

            # For each sample of the data
            for j, sample in enumerate(X_train):

                # Compute the distance between the sample and the centroids
                distances = np.linalg.norm(centroids - sample, axis=1)

                # Assign the sample with the closest centroid
                clusters[j] = np.argmin(distances)

            # Update the centroids
            for z in range(self.n_clusters):
                centroids[z] = X_train[np.where(clusters == z)].mean(axis=0).round(1)

        return clusters

# Part II : An example

# Load the iris data set
iris = datasets.load_iris()

# Define the dataset
X = iris.data

# Shuffle the data
X = X[np.random.permutation(len(X))]

# Machine learning algorithm
kmeans = KMeans(n_clusters=3, max_iter=100)
clusters = kmeans.fit_predict(X)

# Visualize the clusters
fig, ax = plt.subplots()

ax.scatter(X[:, 2], X[:, 3], c=clusters)

plt.show()
