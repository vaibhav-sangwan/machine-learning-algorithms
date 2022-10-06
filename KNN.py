# Features of training data should be fed in following format:
#1) X as a 2D numpy.ndarray of floats or integers
#2) Y as 1D numpy.ndarray with same length as that of X

import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k = 5 ,feature_selection = False):
        self.feature_selection = feature_selection
        self.k = k

    def fit(self, X:np.ndarray, Y:np.ndarray):
        self.X = X.copy()
        self.Y = Y.copy()
        self.features = self.get_features(X, Y)

    def predict_one(self, x):
        distances = []
        for i in range(self.X.shape[0]):
            distance = np.sum(((self.X[i] - x) ** 2))
            index = i
            distances.append([distance, index])
        distances = sorted(distances)
        targets = []
        for i in range(self.k):
            targets.append(self.Y[distances[i][1]])
        return Counter(targets).most_common(1)[0][0]

    def predict(self, X):
        Y  = []
        for i in range(len(X)):
            Y.append(self.predict_one(X[i]))
        return Y

    def score(self, X, Y):
        pass

    def get_features(self, X, Y):
        return [i for i in range(len(Y))]