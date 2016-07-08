from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import operator

data = pd.read_csv('../../data/scores.csv')

y = np.array(data.values[0::, 0])

roc_auc = [0] * 4
for i in range(1, 5):
    p = np.array(data.values[0::, i])
    roc_auc[i-1] = roc_auc_score(y, p)

index, value = max(enumerate(roc_auc), key=operator.itemgetter(1))
column_name = data.columns.values[index+1]

f = open('metrics_q3.txt', 'w')
print(column_name, file=f, end='')
f.close()

precision_recall = [0] * 4
for i in range(1, 5):
    p = np.array(data.values[0::, i])
    [precision, recall, thresholds] = precision_recall_curve(y, p)
    precision_recall[i-1] = max(precision[recall>=0.7])

index, value = max(enumerate(precision_recall), key=operator.itemgetter(1))
column = data.columns.values[index+1]

f = open('metrics_q4.txt', 'w')
print(column, file=f, end='')
f.close()
