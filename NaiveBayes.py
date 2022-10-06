#features of training data should be fed in following format:
#1) X as a pandas dataframe
#2) features having float values will be considered continuous valued features
#3) features having other type values will be considered discrete valued features
#4) Y as numpy.ndarray

import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, Y:np.ndarray):
        classes = set(Y)
        num_features = X.shape[1]
        X.columns = [i for i in range(num_features)]
        result = {}
        result["total data points"] = len(Y)
        for curr_class in classes:
            result[curr_class] = {}
            X_curr_class = X[Y == curr_class]
            Y_curr_class = Y[Y == curr_class]
            result[curr_class]["count"] = len(Y_curr_class)
            for f in range(num_features):
                result[curr_class][f] = {}
                if(X[f].dtype != float):
                    possible_values = set(X_curr_class[f])
                    for value in possible_values:
                        result[curr_class][f][value] = np.array(X_curr_class[f] == value).sum()
                else:
                    result[curr_class][f] = {}
                    result[curr_class][f]["mean"] = np.mean(X_curr_class[f])
                    result[curr_class][f]["std"] = np.std(X_curr_class[f])
        self.dictionary = result

    def get_probability(self, x:pd.DataFrame, curr_class):
        output = np.log(self.dictionary[curr_class]["count"]) -  np.log(self.dictionary["total data points"])
        num_features = len(self.dictionary[curr_class].keys()) - 1
        for f in range(num_features):
            if(not "mean" in self.dictionary[curr_class][f]):
                num = self.dictionary[curr_class][f][x[f]] + 1
                den = self.dictionary[curr_class]["count"] + len(self.dictionary[curr_class][f].keys())
                prob = np.log(num) - np.log(den)
            else:
                mean = self.dictionary[curr_class][f]["mean"]
                std = self.dictionary[curr_class][f]["std"]
                num = - (((x[f] - mean)**2)/(2 * (std ** 2))) * np.log(np.e)
                den = (np.log(2 * np.pi * (std ** 2)))/2
                prob = num - den
            output = output + prob
        return output

    def predict_point(self, x:np):
        max_prob = -1000
        best_class = -1
        first_run = True
        for curr_class in self.dictionary.keys():
            if curr_class == "total data points":
                continue
            prob_curr_class = self.get_probability(x, curr_class)
            if (first_run or (prob_curr_class > max_prob)):
                max_prob = prob_curr_class
                best_class = curr_class
            first_run = False
        return best_class

    def predict(self, X:pd.DataFrame):
        Y = []
        num_features = X.shape[1]
        X.columns = [i for i in range(num_features)]
        for i in range(X.shape[0]):
            Y.append(self.predict_point(X.iloc[i, :]))
        return Y