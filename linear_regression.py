import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

class LR:

    def __init__(self):
        self.m = 0
        self.c = 0

    def fit(self, x:np.array, y:np.array):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        m = (np.mean(x*y) - (x_mean*y_mean))/(np.mean(x*x) - (x_mean*x_mean))
        c = y_mean - (m * x_mean)
        self.m, self.c = m, c
        return (m, c)

    def predict(self, x:np.array):
        y_pred = self.m * x + self.c
        return y_pred

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_mean = y_test.mean()
        u = np.sum((y_test - y_pred)**2)
        v = np.sum((y_test - y_mean)**2)
        scr = 1 - (u / v)
        return scr

    def cost(self, y_pred, y_test):
        error = ((y_test - y_pred)**2).sum()
        return error

lr = LR()
algl = LinearRegression()

x = np.arange(0, 10, 1)
y = np.random.randint(0, 11, size=10)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_test = x_test.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

lr.fit(x_train, y_train)
algl.fit(x_train, y_train)

print(lr.score(x_test, y_test))
print(algl.score(x_test, y_test))

