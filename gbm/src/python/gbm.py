from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import operator

class GBM:
    def __init__(self, n, r):
        self.n = n
        self.clf = GradientBoostingClassifier(n_estimators=n, learning_rate=r, verbose=False, random_state=241)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def log_loss(self, X, y):
        loss = [0] * self.n
        for i, proba in zip(
                range(0, self.n),
                self.clf.staged_predict_proba(X)):
            loss[i] = log_loss(y, proba)
        return loss

data = pd.read_csv('../../data/gbm-data.csv')
data_sample = data.head(n=10)
data_sample.to_csv('../../data/gbm-data-sample.csv', index=False)

X = np.array(data.values[0::, 1::])
y = np.array(data.values[0::, 0])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.8, random_state=241)

clf = GBM(250, 0.2)
clf.fit(X_train, y_train)
loss = clf.log_loss(X_test, y_test)

index, gbm_loss = min(enumerate(loss), key=operator.itemgetter(1))

f = open('gbm_q1.txt', 'w')
print("overfitting", file=f, end='')
f.close()

f = open('gbm_q2.txt', 'w')
print("%.2f" % gbm_loss, index+1, file=f, end='')
f.close()

clf = RandomForestClassifier(n_estimators=index+1, random_state=241)
clf.fit(X_train, y_train)
forest_loss = log_loss(y_test, clf.predict_proba(X_test))

f = open('gbm_q3.txt', 'w')
print("%.2f" % forest_loss, file=f, end='')
f.close()
