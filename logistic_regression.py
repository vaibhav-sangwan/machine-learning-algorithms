import numpy as np
from sklearn import datasets
from sklearn import model_selection

from sklearn.metrics import confusion_matrix

class GradientDescent:

    def __init__(self):
        pass
    
    @classmethod
    def gd(cls, X, Y, learning_rate, num_itr):
        M = X.shape[0]
        N = X.shape[1] + 1
        m = np.full(N, 0)
        X = np.append(X, np.ones(M).reshape(-1,1), axis = 1)
        for i in range(num_itr):
            m = cls.step_gradient(X, Y, learning_rate, m)
            #print(cls.cost(X, Y, m))
        return m[:-1], m[-1]

    @classmethod
    def step_gradient(cls, X, Y, learning_rate, m):
        M = X.shape[0]
        N = X.shape[1]
        m_slope = np.zeros(N)
        Y_pred = cls.findhx(X, m)
        for i in range(M):
            hx = Y_pred[i]
            for j in range(N):
                m_slope[j] += (-1/M) * (Y[i] - hx) * X[i, j]
        m_dash = m - (learning_rate * m_slope)
        return m_dash

    @classmethod
    def cost(cls, X:np.ndarray, Y:np.ndarray, m:np.ndarray):
        error = 0
        M = X.shape[0]
        N = X.shape[1]
        Y_pred = cls.findhx(X, m)
        for i in range(M):
            y = Y[i]
            hx = Y_pred[i]
            error += (1/M) * ((-y * np.log(hx))-((1 - y) * (np.log(1 - hx))))
        return (error.real, error.imag)

    @classmethod
    def findhx(cls, X, m):
        M = X.shape[0]
        Y = np.zeros(M)
        for i in range(M):
            mtx = np.sum(m * X[i])
            Y[i] = 1 / (1 + (np.e ** (-mtx)))
        return Y



class LogRegression:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.m, self.c = GradientDescent.gd(X, Y, 0.000001, 1000)

    def predict(self, X:np.ndarray):
        Y = self.predict_proba(X)
        Y[Y > 0.5] = 1.0
        Y[Y <= 0.5] = 0.0
        return Y

    def predict_proba(self, X:np.ndarray):
        M = X.shape[0]
        m_x = self.m * X
        mtx = np.zeros(M)
        for i in range(M):
            mtx[i] = sum(m_x[i]) + self.c
        Y  = 1 / (1 + (np.e ** (-mtx)))
        return Y

    def score(self, X:np.ndarray, Y:np.ndarray):
        Y_pred = self.predict(X)
        M = X.shape[0]
        accurate = len(Y[Y_pred == Y])
        scr = accurate / M
        return scr

cancer_data = datasets.load_breast_cancer()
X = np.array(cancer_data['data'])
Y = np.array(cancer_data['target'])
print(X)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)
alg1 = LogRegression()
alg1.fit(X_train, Y_train)
y_test_pred = alg1.predict(X_test)
y_train_pred = alg1.predict(X_train)
train_score = alg1.score(X_train, Y_train)
test_score = alg1.score(X_test, Y_test)
print("Train Score", train_score, "\nTest Score", test_score)
print("Train data confusion matrix:")
print(confusion_matrix(Y_train, y_train_pred))
print("Test data confusion matrix:")
print(confusion_matrix(Y_test, y_test_pred))