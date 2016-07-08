from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.preprocessing import scale
import operator

class Classifier:
    def __init__(self, n):
        self.n = n

    def fit(self, X, y):
        means = []
        for k in range(1, self.n+1):
            kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_validation.cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
            mean = scores.mean()
            means.append(mean)

        index, value = max(enumerate(means), key=operator.itemgetter(1))
        self.k = index+1
        self.quality = value

data = pd.read_csv('../../data/wine.data', header=None)

X = np.array(data.values[0::, 1::])
y = np.array(data.values[0::, 0])

clf = Classifier(n=50)
clf.fit(X, y)

f = open('wine_q1.txt', 'w')
print(clf.k, file=f, end='')
f.close()

f = open('wine_q2.txt', 'w')
print("%.2f" % clf.quality, file=f, end='')
f.close()

X_std = scale(X)
clf = Classifier(n=50)
clf.fit(X_std, y)

f = open('wine_q3.txt', 'w')
print(clf.k, file=f, end='')
f.close()

f = open('wine_q4.txt', 'w')
print("%.2f" % clf.quality, file=f, end='')
f.close()
