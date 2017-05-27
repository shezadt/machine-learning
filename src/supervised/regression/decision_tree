#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Load the libraries
from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Part I : Algorithm

# Define a class that represents a decision node or a terminal leaf
class DecisionNode():

    def __init__(self, feature_index=None, threshold=None, value=None,
                 left_branch=None, right_branch=None):

        # Index of the feature tested
        self.feature_index = feature_index

        # Threshold value of the feature tested
        self.threshold = threshold

        # The value if the node is a terminal leaf
        self.value = value

        # The left subtree
        self.left_branch = left_branch

        # The right subtree
        self.right_branch = right_branch

# The algorithm
class DecisionTreeRegressor():

    def __init__(self, max_depth=float('inf'), min_samples_split=2,
                 min_impurity_split=1e-7):

        # Root node
        self.root = None

        # The maximum depth of the tree
        self.max_depth = max_depth

        # The minimum of samples required to make a split
        self.min_samples_split = min_samples_split

        # The minimum of impurity required to make a split
        self.min_impurity_split = min_impurity_split

    def _divide_on_feature(self, X, feature_index, threshold):

        # Get the index of samples that are above the threshold
        split_index = X[:, feature_index] >= threshold

        # Split the data based on this threshold
        X_left = X[np.invert(split_index)]
        X_right = X[split_index]

        return X_left, X_right

    def _variance_reduction(self, y, y_left, y_right):

        # Total variance
        var_tot = y.var()

        # Variance of the left data
        var_left = y_left.var()
        prop_left = len(y_left) / len(y)

        # Variance of the right data
        var_right = y_right.var()
        prop_right = len(y_right) / len(y)

        # Variance reduced
        var_reduced = var_tot - (prop_left * var_left + prop_right * var_right)

        return var_reduced

    def _build_tree(self, X_train, y_train, current_depth=0):

        # The impurity
        largest_impurity = 0

        # A dictionary with the feature index and its threshold
        best_split_feature = None

        # A dictionary with the left subtree and the right subtree
        best_subtrees = None

        # Add a dimension to the target if needed
        if len(np.shape(y_train)) == 1:
            y = np.expand_dims(y_train, axis=1)
        else:
            y = y_train

        # Concatenate X_train and y_train
        X_y = np.concatenate((X_train, y), axis=1)

        # Get the dimension of the space
        n_samples, n_features = np.shape(X_train)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:

            # Compute the impurity for each feature
            for feature_index in range(n_features):

                # Get the unique values of that feature
                feature_values = X_train[:, feature_index]
                unique_values = np.unique(feature_values)

                # Split the data for each value
                for threshold in unique_values:
                    X_y_left, X_y_right = self._divide_on_feature(X_y,
                                                              feature_index,
                                                              threshold)

                    # If the split ends with data on the left and on the right
                    if len(X_y_left) > 0 and len(X_y_right) > 0:

                        # Get the left and the right target
                        y_left = X_y_left[:, n_features:]
                        y_right = X_y_right[:, n_features:]

                        # Get the reduced variance
                        impurity = self._variance_reduction(y_train, y_left,
                                                            y_right)

                        # If the split gives a better reduced variance save it
                        if impurity > largest_impurity:

                            # Save the impurity
                            largest_impurity = impurity

                            # Save the best split
                            best_split_feature = {'feature_index' : feature_index,
                                                  'threshold' : threshold}

                            # Save the best subtrees
                            best_subtrees = {'X_left' : X_y_left[:, :n_features],
                                             'y_left' : X_y_left[:, n_features:],
                                             'X_right' : X_y_right[:, :n_features],
                                             'y_right' : X_y_right[:, n_features:]}


        # If the split is still not pure enough build a left branch and a right
        # branch
        if largest_impurity > self.min_impurity_split:

            # Build the left branch
            left_branch = self._build_tree(best_subtrees['X_left'],
                                            best_subtrees['y_left'],
                                            current_depth + 1)

            # Build the right branch
            right_branch = self._build_tree(best_subtrees['X_right'],
                                            best_subtrees['y_right'],
                                            current_depth + 1)

            # Build the decision node at this depth
            return DecisionNode(feature_index=best_split_feature['feature_index'],
                                threshold=best_split_feature['threshold'],
                                right_branch=right_branch,
                                left_branch=left_branch)

        # Otherwise we are at a leaf so we can get the value
        leaf_value = y.mean()

        return DecisionNode(value=leaf_value)

    def fit(self, X_train, y_train):

        self.root = self._build_tree(X_train, y_train)

    # Do a recursive search down the tree and make a prediction of the sample by
    # the value of the leaf that we end up at
    def predict_value(self, sample, tree=None):

        # Start by the root node
        if tree is None:
            tree = self.root

        # Get the value if we have a prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = sample[tree.feature_index]

        # Determine if we will follow left or right branch
        if feature_value >= tree.threshold:
            branch = tree.right_branch
        else:
            branch = tree.left_branch

        # Go through the subtree
        return self.predict_value(sample, branch)

    # Predict for the test set
    def predict(self, X_test):

        # Define the prediction vector
        y_pred = np.zeros(len(X_test))

        # Predict for each test sample
        for i, test_sample in enumerate(X_test):

            y_pred[i] = self.predict_value(test_sample)

        return y_pred

# Part II : An example

# Load the Boston data set
boston = datasets.load_boston()

# Define the features
X = boston.data
y = boston.target

# Define the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Machine learning algorithm
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Compute the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Compute the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Visualize the results
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Decision tree on the Boston data set')
plt.xlabel('Real values')
plt.ylabel('Prediction')
plt.show()
