from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('../../data/perceptron-train.csv', header=None)
test = pd.read_csv('../../data/perceptron-test.csv', header=None)

X_train = np.array(train.values[0::, 1::])
y_train = np.array(train.values[0::, 0])

X_test = np.array(test.values[0::, 1::])
y_test = np.array(test.values[0::, 0])

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train)
score_scaled = clf.score(X_test_scaled, y_test)

accuracy = score_scaled - score

f = open('perceptron.txt', 'w')
print("%.3f" % accuracy, file=f, end='')
f.close()
