from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pd.read_csv('../../data/classification.csv')

y = np.array(data.values[0::, 0])
p = np.array(data.values[0::, 1])

TP = 0
FP = 0
FN = 0
TN = 0

for i in range(0, len(y)):
    TP += 1 if y[i] == 1 and p[i] == 1 else 0
    FN += 1 if y[i] == 1 and p[i] == 0 else 0
    FP += 1 if y[i] == 0 and p[i] == 1 else 0
    TN += 1 if y[i] == 0 and p[i] == 0 else 0

f = open('metrics_q1.txt', 'w')
print(TP, FP, FN, TN, file=f, end='')
f.close()

accuracy = accuracy_score(y, p)
precision = precision_score(y, p)
recall = recall_score(y, p)
f1 = f1_score(y, p)

f = open('metrics_q2.txt', 'w')
print("%.2f" % accuracy, "%.2f" % precision, "%.2f" % recall, "%.2f" % f1, file=f, end='')
f.close()
