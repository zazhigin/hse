from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.svm import SVC

data = pd.read_csv('../../data/svm-data.csv', header=None)

X = np.array(data.values[0::, 1::])
y = np.array(data.values[0::, 0])

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)

indexes = clf.support_+1
result = ' '.join(str(x) for x in indexes)

f = open('svm.txt', 'w')
print(result, file=f, end='')
f.close()
