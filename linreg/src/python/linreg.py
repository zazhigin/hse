from __future__ import print_function
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

class SalaryRegression:
    def __init__(self):
        self.vect = TfidfVectorizer(min_df=5)
        self.enc1 = DictVectorizer()
        self.enc2 = DictVectorizer()
        self.reg = Ridge(alpha=1, random_state=241)

    def get_X(self, data, fit_transform):
        # FullDescription
        description = data['FullDescription'].str
        description = description.lower()
        description = description.replace('[^a-zA-Z0-9]', ' ', regex=True)
        if fit_transform:
            description = self.vect.fit_transform(description)
        else:
            description = self.vect.transform(description)

        # LocationNormalized
        data['LocationNormalized'].fillna('nan', inplace=True)
        location = data[['LocationNormalized']].to_dict('records')
        if fit_transform:
            location = self.enc1.fit_transform(location)
        else:
            location = self.enc1.transform(location)

        # ContractTime
        data['ContractTime'].fillna('nan', inplace=True)
        time = data[['ContractTime']].to_dict('records')
        if fit_transform:
            time = self.enc2.fit_transform(time)
        else:
            time = self.enc2.transform(time)

        return hstack([description, location, time])

    def get_Y(self, data):
        return data['SalaryNormalized']

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

data_train = pd.read_csv('../../data/salary-train.csv')
data_train_sample = data_train.head(n=10)
data_train_sample.to_csv('../../data/salary-train-sample.csv', index=False)

salary = SalaryRegression()
X_train = salary.get_X(data_train, True)
y_train = salary.get_Y(data_train)
salary.fit(X_train, y_train)

data_test = pd.read_csv('../../data/salary-test-mini.csv')

X_test = salary.get_X(data_test, False)
y_test = salary.predict(X_test)

f = open('linreg.txt', 'w')
print("%.2f" % y_test[0], "%.2f" % y_test[1], file=f, end='')
f.close()
