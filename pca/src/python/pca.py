from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv('../../data/close_prices.csv')
djia = pd.read_csv('../../data/djia_index.csv')['^DJI']

X = np.array(data.values[0::, 1::])

pca = PCA(n_components=10)
pca.fit(X)

ratio = pca.explained_variance_ratio_

std = 0
for i in range(0, 10):
   std += ratio[i]
   if std >= 0.90: break

f = open('pca_q1.txt', 'w')
print(i+1, file=f, end='')
f.close()

XX = pca.transform(X)
x0 = map(lambda x: x[0], XX)
corr = np.corrcoef(x0, djia)[0][1]

f = open('pca_q2.txt', 'w')
print("%.2f" % corr, file=f, end='')
f.close()

components = pca.components_[0]
index = np.where(components == max(components))[0][0]
column_name = data.columns[index+1]

f = open('pca_q3.txt', 'w')
print(column_name, file=f, end='')
f.close()
