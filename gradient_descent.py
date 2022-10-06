import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class GradientDescent:
    def __init__(self):
        pass

    def gd(self, X:np.ndarray, Y:np.ndarray, learning_rate, num_iterations):
        M = X.shape[0]
        N = X.shape[1] + 1
        m = np.zeros(N)
        X = np.append(X, np.ones(M).reshape(-1,1), axis = 1)
        for i in range(num_iterations):
            m = self.step_gradient(X, Y, learning_rate, m)
            print(self.cost(X, Y, m))
        return m[:-1], m[-1]

    def step_gradient(self, X:np.ndarray, Y:np.ndarray, learning_rate, m:np.ndarray):
        M = X.shape[0]
        N = X.shape[1]
        m_slope = np.zeros(N)
        for i in range(M):
            pred_value = np.sum(m * X[i])
            for j in range(N):
                m_slope[j] += (-2/M) * (Y[i] - pred_value) * X[i, j]
        m_dash = m - (learning_rate * m_slope)
        return m_dash

    def cost(self, X:np.ndarray, Y:np.ndarray, m:np.ndarray):
        cost = 0
        M = X.shape[0]
        for i in range(M):
            cost += (1/M) * ((Y[i] - np.sum(X[i] * m))**2)
        return cost

fl = pd.read_csv("train.csv")
X_train = fl.iloc[:, :-1]
Y_train = fl.iloc[:, -1]

scaler = StandardScaler()
scaler.fit(X_train)#feature scaling on training data set, we don't need to scale output
scaled_X_train = scaler.transform(X_train)

grad = GradientDescent()
m, c = grad.gd(scaled_X_train, Y_train, 0.1, 100)#performing gradient descent on scaled data to get m-array and c

X_test = np.loadtxt(r"test.csv",delimiter = ',')
M = X_test.shape[0]
Y_pred = np.zeros(M)
scaled_X_test = scaler.transform(X_test)#feature scaling on testing data set
for i in range(M):
    Y_pred[i] = np.sum(m * scaled_X_test[i]) + c
print(Y_pred)

alg1 = LinearRegression()   #comparing against in built python algorithm
alg1.fit(X_train, Y_train)
y_linear_pred = alg1.predict(X_test)

plt.scatter(y_linear_pred, Y_pred, color = "r")
plt.plot([400, 500], [400, 500], "b--")
plt.show()