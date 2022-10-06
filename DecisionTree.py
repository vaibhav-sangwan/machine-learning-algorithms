#features of training data should be fed in following format:
#1) X as a pandas dataframe
#2) features having float values will be considered continuous valued features
#3) features having other type values will be considered discrete valued features
#4) Y as numpy.ndarray

import numpy as np

class Node:
    def __init__(self, X, Y, features):
        self.size = len(Y)
        self.classes = {i:len(Y[Y == i]) for i in set(Y)}
        self.return_value = max(self.classes, key=self.classes.get)
        self.childs = dict()
        self.is_leaf = (len(set(Y)) == 1)        
        self.f_best_val = -1
        self.f = self.find_feature_to_split_upon(X, Y, features) #feature to split upon
        if self.f != -1:
            self.is_f_continuous = (X[self.f].dtype == float)
        else:
            self.is_f_continuous = False

    def find_feature_to_split_upon(self, X, Y, features):
        max_accuracy = self.return_value / self.size
        f = -1
        if len(features) == 0:
            return f
        for i in features:           
            correct_pred = 0
            if(X[i].dtype != float):
                for value in set(X[i]):
                    Y_child = Y[X[i] == value]
                    Y_values, counts = np.unique(Y_child, return_counts=True)
                    correct_pred += np.max(counts)
                accuracy = correct_pred / self.size
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    f = i
                    self.is_f_continuous = False
            else:
                sorted_values = list(set(X[i].sort_values()))
                points = []
                for index in range(0, len(sorted_values) - 1):
                    points.append((sorted_values[index] + sorted_values[index + 1])/2)
                for point in points:
                    Y_0 = Y[X[i] < point]
                    Y_1 = Y[X[i] >= point]
                    Y_values, counts = np.unique(Y_0, return_counts=True)
                    correct_pred += np.max(counts)
                    Y_values, counts = np.unique(Y_1, return_counts=True)
                    correct_pred += np.max(counts)
                    accuracy = correct_pred / self.size
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        f = i
                        self.is_f_continuous = True
                        self.f_best_val = point                  
        return f

class DecisionTree:
    def fit(self, X, Y):
        features = [i for i in range(X.shape[1])]
        X.columns = features
        root = Node(X, Y, features)
        self.make_decision_tree(X, Y, features, root)
        self.root = root
        self.out_type = type(Y)

    def predict(self, X):
        M = X.shape[0]
        Y = np.zeros(M, dtype=self.out_type)
        for i in range(M):
            curr = self.root
            while(not curr.is_leaf):
                if(not curr.is_f_continuous):
                    if X.iloc[i, curr.f] in curr.childs:
                        curr = curr.childs[X.iloc[i, curr.f]]
                    else:
                        break
                else:
                    if (X.iloc[i, curr.f] < curr.f_best_val) in curr.childs:
                        curr = curr.childs[X.iloc[i, curr.f] < curr.f_best_val]
                    else:
                        break
            Y[i] = curr.return_value
        return Y

    def make_decision_tree(self, X, Y, features,  node:Node):
        if(node.is_leaf or len(features) == 0):
            node.is_leaf = True
        else:
            f = node.f
            new_features = features.copy()
            new_features.remove(f)
            if(node.is_f_continuous == False):
                for i in set(X[f]):
                    X_child = X[X[f] == i]
                    Y_Child = Y[X[f] == i]
                    child_node = Node(X_child, Y_Child, new_features)
                    node.childs[i] = child_node
                    self.make_decision_tree(X_child, Y_Child, new_features, child_node)
            else:
                X_0 = X[X[f] <= node.f_best_val]
                Y_0 = Y[X[f] <= node.f_best_val]
                X_1 = X[X[f] > node.f_best_val]
                Y_1 = Y[X[f] > node.f_best_val]
                child_node_1 = Node(X_0, Y_0, new_features)
                node.childs[True] = child_node_1
                self.make_decision_tree(X_0, Y_0, new_features, child_node_1)
                child_node_2 = Node(X_1, Y_1, new_features)
                node.childs[False] = child_node_2
                self.make_decision_tree(X_1, Y_1, new_features, child_node_2)
                
    def print_tree(self):
        q = [self.root]
        while len(q) != 0:
            curr = q[0]
            self.print_node(curr)
            print("\n\n")
            q.extend(curr.childs.values())
            q.pop(0)
    
    def print_node(self, node:Node):
        print("size", node.size)
        print("classes", node.classes)
        print("majority vote", node.return_value)
        print("is a Leaf:", node.is_leaf)
        print("feature to split upon", node.f)
        print("number of childeren:", len(node.childs))
        print("childs", node.childs)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        correct_pred = 0
        for i in range(len(Y)):
            if Y_pred[i] == Y[i]:
                correct_pred += 1
        score = correct_pred / len(Y)
        return score