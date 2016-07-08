from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../../data/titanic.csv', index_col='PassengerId')

# Q1. Number of male & female
f1 = open('titanic_q1.txt', 'w')
q1_sex = data['Sex'].value_counts()
print(q1_sex.get('male'), q1_sex.get('female'), file=f1, end='')
f1.close()

# Q2. Survived percent
f2 = open('titanic_q2.txt', 'w')
q2_survived = data['Survived'].value_counts()
q2_percent = float(q2_survived.get(1)) / data['Survived'].count() * 100
print("%.2f" % q2_percent, file=f2, end='')
f2.close()

# Q3. First class percent
f3 = open('titanic_q3.txt', 'w')
q3_pclass = data['Pclass'].value_counts()
q3_percent = float(q3_pclass.get(1)) / data['Pclass'].count() * 100
print("%.2f" % q3_percent, file=f3, end='')
f3.close()

# Q4. Age average & median
f4 = open('titanic_q4.txt', 'w')
q4_mean = data['Age'].mean()
q4_median = data['Age'].median()
print("%.2f" % q4_mean, "%.2f" % q4_median, file=f4, end='')
f4.close()

# Q5. SibSp & Parch correlation
f5 = open('titanic_q5.txt', 'w')
q5_corr = data.corr('pearson')['SibSp']['Parch']
print("%.2f" % q5_corr, file=f5, end='')
f5.close()

# Q6. Most popular female name
f6 = open('titanic_q6.txt', 'w')
q6_sex = data.set_index('Sex')
q6_female = q6_sex.loc["female"]
q6_female = q6_female['Name'].replace("[\w '-]+, Mrs\. \w+ [\w ]*\((\w+).*", "\\1")
q6_female = q6_female.replace("[\w '-]+, (Mrs\.|Miss\.|Lady\.|the Countess\. of) \((\w+).*", "\\2")
q6_female = q6_female.replace("[\w '-]+, (Mrs|Miss|Mme|Ms|Mlle)\. (\w+).*", "\\2")
q6_name = q6_female.value_counts().index[0]
print(q6_name, file=f6, end='')
f6.close()

# Q7. Features importance
f7 = open('titanic_q7.txt', 'w')
data = pd.read_csv('../../data/titanic.csv')
data['Gender'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
data = data[~np.isnan(data.Age)]
data = data.drop(['Name', 'Sex', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

X = np.array(data.values[0::, 1::])
y = np.array(data.values[0::, 0])

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_
features = data.columns.values[1::]
features[features == 'Gender'] = 'Sex'

result = pd.DataFrame( {'Feature' : features, 'Importance' : importances} )
result = result.sort_values('Importance', ascending=0).set_index('Feature')

print(result.index[0], result.index[1], file=f7, sep=', ', end='')
f7.close()
