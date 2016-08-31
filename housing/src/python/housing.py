from __future__ import print_function
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class Regression:
    def fit(self, X, y):
        means = {}
        for p in np.linspace(1, 10, num=200):
            kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
            knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
            scores = cross_val_score(knn, X, y, cv=kf, scoring='mean_squared_error')
            means[p] = scores.mean()
        return max(means, key=means.get)

boston = datasets.load_boston()

X = scale(boston.data)
y = boston.target

reg = Regression()
p = reg.fit(X, y)

f = open('housing.txt', 'w')
print("%.1f" % p, file=f, end='')
f.close()
