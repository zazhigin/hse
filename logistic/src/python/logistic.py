from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from math import exp, sqrt

data = pd.read_csv('../../data/data-logistic.csv', header=None)

X = np.array(data.values[0::, 1::])
y = np.array(data.values[0::, 0])

class LogisticRegression:
    def __init__(self, k, C, e, n):
        self.k = k
        self.C = C
        self.e = e
        self.n = n

    def fit(self, X, y):
        w = [0] * len(X[0])
        for iter in range(0, self.n):
            ww = [0] * len(w)
            for index in range(0, len(w)):
                ww[index] = self.calc(X, y, w, index)
            if self.dist(w, ww) < self.e:
                break
            w = ww
        self.w = w
        self.iter = iter

    def calc(self, X, y, w, index):
        summ = 0.0
        for i in range(0, len(X)):
            sigma = 0.0
            for ii in range(0, len(w)):
                sigma += w[ii] * X[i][ii]
            summ += y[i] * X[i][index] * (1 - 1 / (1 + exp(-y[i] * sigma)))
        return w[index] + (self.k / len(X)) * summ - self.k * self.C * w[index]

    def dist(self, a, b):
        d = 0
        for i in range(0, len(a)):
            d += (a[i] - b[i])**2
        return sqrt(d)

    def auc_roc(self, X, y):
        return roc_auc_score(y, map(lambda x: self.prob(self.w, x), X))

    def prob(self, x, w):
        M = 0
        for i in range(0, len(x)):
            M += w[i] * x[i]
        return 1 / (1 + exp(-M))

def logistic(k, C, e, n):
    lr = LogisticRegression(k, C, e, n)
    lr.fit(X, y)
    print('Convergence on', lr.iter, 'step: C =', C)
    return lr.auc_roc(X, y)

auc_roc    = logistic(k=0.1, C=0,  e=0.00001, n=10000)
auc_roc_l2 = logistic(k=0.1, C=10, e=0.00001, n=10000)

f = open('logistic.txt', 'w')
print("%.3f" % auc_roc, "%.3f" % auc_roc_l2, file=f, end='')
f.close()
