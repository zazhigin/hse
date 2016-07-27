from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

class Forest:
    def __init__(self, quality):
        self.quality = quality

    def fit(self, X, y):
        for n in range(1, 51):
            kf = KFold(len(X), n_folds=5, shuffle=True, random_state=1)
            clf = RandomForestRegressor(n_estimators=n, random_state=1)
            scores = cross_val_score(clf, X, y, scoring='r2', cv=kf)
            mean = scores.mean()
            if mean > self.quality:
                return n

data = pd.read_csv('../../data/abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x =='M' else (-1 if x == 'F' else 0))

X = np.array(data.values[0::, 0:8])
y = np.array(data.values[0::, 8])

forest = Forest(quality=0.52)
n = forest.fit(X, y)

f = open('forest.txt', 'w')
print(n, file=f, end='')
f.close()
