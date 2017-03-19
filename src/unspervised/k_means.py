#!/usr/bin/env python2
# -*- coding: utf-8 -*-

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

# Define the number of clusters
K = 3

# Define the number of iterations
N_iter = 100

# Get the number of observations
N_obs = len(X)

# Shuffle the data
X = X[np.random.permutation(N_obs)]

# Initialize the centroids as random samples from the data
centroids = X[np.random.randint(N_obs, size=K)]

# Define the clusters vector
clusters = np.zeros(N_obs)

# Loop on the number of iterations
for i in range(N_iter):
    
    # For each sample of the data
    for j, sample in enumerate(X):
        
        # Define a distance vector
        distances = np.zeros(K)
        
        # Compute the distance between the sample and the centroids
        for z, centroid in enumerate(centroids):
            distances[z] = np.linalg.norm(sample - centroid)
        
        # Assign the sample with the closest centroid
        clusters[j] = np.argmin(distances)
    
    # Update the centroids
    for z in range(K):
        centroids[z] = X[np.where(clusters == z)].mean(axis=0).round(1)

# Visualize the clusters
class_0 = X[np.where(clusters == 0)]
class_1 = X[np.where(clusters == 1)]
class_2 = X[np.where(clusters == 2)]

plt.scatter(class_0[:,2], class_0[:,3], color = 'blue')
plt.scatter(class_1[:,2], class_1[:,3], color = 'red')
plt.scatter(class_2[:,2], class_2[:,3], color = 'green')
plt.legend(('Class 0', 'Class 1', 'Class 2'), scatterpoints=1, loc=2)
plt.show()
