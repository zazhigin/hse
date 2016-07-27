from __future__ import print_function
from sklearn.cross_validation import KFold, cross_val_predict, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from datetime import datetime
import pandas as pd
import numpy as np
import operator

class Dota2:
    def getFeatures(self, with_categorial=False):
        features_list = []

        features_list.append('lobby_type') if with_categorial else {}

        features_list += self.getFeaturesForHeroes(with_categorial)
        features_list += self.getFirstBloodFeatures()
        features_list += self.getFeaturesForTeams()

        return features_list

    def getFeaturesForHeroes(self, with_categorial):
        features = []
        for x in ['r1', 'r2', 'r3', 'r4', 'r5', 'd1', 'd2', 'd3', 'd4', 'd5']:
            features.append(x + '_hero')  if with_categorial else {}
            features.append(x + '_level')
            features.append(x + '_xp')
            features.append(x + '_gold')
            features.append(x + '_lh')
            features.append(x + '_kills')
            features.append(x + '_deaths')
            features.append(x + '_items')
        return features

    def getFeaturesWithHeroes(self):
        features = [
            'r1_hero','r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
            'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
        return features

    def getFirstBloodFeatures(self):
        features = [
            'first_blood_time',
            'first_blood_team',
            'first_blood_player1',
            'first_blood_player2']
        return features

    def getFeaturesForTeams(self):
        features = []
        for x in ['radiant', 'dire']:
            features.append(x + '_bottle_time')
            features.append(x + '_courier_time')
            features.append(x + '_flying_courier_time')
            features.append(x + '_tpscroll_count')
            features.append(x + '_boots_count')
            features.append(x + '_ward_observer_count')
            features.append(x + '_ward_sentry_count')
            features.append(x + '_first_ward_time')
        return features


# First part: Gradient boosting

class Classifier:
    def __init__(self, max_n, max_depth=3):
        self.max_n = max_n
        self.max_depth = max_depth

    def fit(self, X, y):
        for n in range(10, self.max_n+10, 10):
            start_time = datetime.now()
            kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=42)
            clf = GradientBoostingClassifier(n_estimators=n, max_depth=self.max_depth)
            probas = cross_val_predict(clf, X, y, cv=kf)
            score = roc_auc_score(y, probas)
            elapsed_time = datetime.now() - start_time
            print('Time elapsed:', elapsed_time.seconds, 'seconds  max_depth =', self.max_depth, ' n_estimators =', n, ' ROC_AUC = %.4f' % score)

data = pd.read_csv('../../data/features.csv', index_col='match_id')

data_sample = data.head(n=10)
data_sample.to_csv('../../data/features-sample.csv', index=True)

# Features with missing data
#features_count = data.count()
#print(features_count[features_count < data.shape[0]])

data.fillna(0, inplace=True)

X = data[Dota2().getFeatures(with_categorial=True)]
y = data['radiant_win']

# Gradient boosting classification
clf = Classifier(max_n=30)
clf.fit(X, y)

# Train optimization time var.1
clf = Classifier(max_n=50, max_depth=2)
clf.fit(X, y)

# Train optimization time var.2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
clf = Classifier(max_n=30)
clf.fit(X_train, y_train)


# Second part: Logistic regression

class Regression:
    def __init__(self, logspace):
        self.logspace=logspace

    def fit(self, X, y):
        scores = {}
        for C in self.logspace:
            start_time = datetime.now()
            kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=42)
            clf = LogisticRegression(penalty='l2', C=C)
            probas = cross_val_predict(clf, X, y, cv=kf)
            scores[C] = roc_auc_score(y, probas)
            elapsed_time = datetime.now() - start_time
            print('Time elapsed:', elapsed_time.seconds, 'seconds  C =', C, ' ROC_AUC = %.5f' % scores[C])
        return scores

X_std = scale(X)

clf = Regression(logspace=np.logspace(-4, 0, num=5))
scores = clf.fit(X_std, y)

C, max_score = max(scores.iteritems(), key=operator.itemgetter(1))
print('with categorial C=', C, ' max_score=',max_score)

X = data[Dota2().getFeatures(with_categorial=False)]
y = data['radiant_win']

X_std = scale(X)

clf = Regression(logspace=np.logspace(-4, 0, num=5))
scores = clf.fit(X_std, y)

C, max_score = max(scores.iteritems(), key=operator.itemgetter(1))
print('no categorial C=', C, ' max_score=',max_score)

heroes = set()
for feature in Dota2().getFeaturesWithHeroes():
    heroes.update(data[feature].value_counts().index.values)

unique_heroes = list(heroes)

print("Unique heroes =", len(unique_heroes))

def getWordsBag(data, unique_heroes):
    X_pick = np.zeros((data.shape[0], len(unique_heroes)))
    for i, match_id in enumerate(data.index):
        for p in xrange(5):
            X_pick[i, unique_heroes.index(data.ix[match_id, 'r%d_hero' % (p+1)])] = 1
            X_pick[i, unique_heroes.index(data.ix[match_id, 'd%d_hero' % (p+1)])] = -1
    return X_pick

X_pick = getWordsBag(data, unique_heroes)
X_std = np.hstack((X_std, X_pick))

clf = Regression(logspace=np.logspace(0, 4, num=5))
scores = clf.fit(X_std, y)

C, max_score = max(scores.iteritems(), key=operator.itemgetter(1))
print('with words bag C=', C, ' max_score=',max_score)

data_test = pd.read_csv('../../data/features_test.csv', index_col='match_id')

data_test_sample = data_test.head(n=10)
data_test_sample.to_csv('../../data/features_test-sample.csv', index=True)

data_test.fillna(0, inplace=True)

X_test = data_test[Dota2().getFeatures(with_categorial=False)]
X_test_std = scale(X_test)

X_test_pick = getWordsBag(data_test, unique_heroes)
X_test_std = np.hstack((X_test_std, X_test_pick))

clf = LogisticRegression(penalty='l2', C=1000)
clf.fit(X_std, y)

p = clf.predict(X_test_std)

df = pd.DataFrame(index=X_test.index, columns=['radiant_win'])
df['radiant_win'] = clf.predict_proba(X_test_std)
df.to_csv('predictions_test.csv')

print("Minimum proba = ", min(df['radiant_win']))
print("Maximum proba = ", max(df['radiant_win']))
