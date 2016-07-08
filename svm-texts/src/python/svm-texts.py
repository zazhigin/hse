from __future__ import print_function
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from heapq import nlargest
import numpy as np

class TextsSVC():
    def __init__(self, grid):
        self.grid = grid

    def fit(self, X, y):
        kf = KFold(len(y), n_folds=5, shuffle=True, random_state=241)
        clf = SVC(kernel='linear', random_state=241)
        gs = GridSearchCV(clf, self.grid, scoring='accuracy', cv=kf)
        gs.fit(X, y)
        scores = map(lambda x: x.mean_validation_score, gs.grid_scores_)
        index = scores.index(max(scores))
        return gs.grid_scores_[index].parameters['C']

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vect = TfidfVectorizer()

X = newsgroups.data
y = newsgroups.target

tfidf = vect.fit_transform(X, y)

grid = {'C': np.power(10.0, np.arange(-5, 6))}

clf = TextsSVC(grid=grid)
C = clf.fit(tfidf, y)

clf = SVC(C=C, kernel='linear', random_state=241)
clf.fit(tfidf, y)

coef = clf.coef_.toarray()[0]
coef = abs(coef)

feature_names = vect.get_feature_names()

words = []
for x in nlargest(10, coef):
    words.append(feature_names[np.where(coef == x)[0][0]])
words.sort()

f = open('svm-texts.txt', 'w')
print(' '.join(words), file=f, end='')
f.close()
