# hse
Higher School of Economics

### Summary table

| Project | Dataset | Features | Size | Scaled | Fillna | Zero | Algorithm | Cross-validation | Optimization | Metrics |
| :------ | :------ | :------- | :--- | :----- | :----- | :--- | :-------- | :--------------- | :----------- | :------ |
| [titanic](https://github.com/zazhigin/hse/tree/master/titanic/src/python) | Titanic | 4 | 891 (714) | no | remove | 9.38% | DecisionTreeClassifier |  |  | feature_importances |
| [wine](https://github.com/zazhigin/hse/tree/master/wine/src/python) | Wine | 13 | 178 | scale | no | 0% | KNeighborsClassifier | KFold(5) | n_neighbors=1/51 | Accuracy |
| [housing](https://github.com/zazhigin/hse/tree/master/housing/src/python) | Boston | 13 | 506 | no | no | 0% | KNeighborsRegressor | KFold(5) | p | MSE |
| [perceptron](https://github.com/zazhigin/hse/tree/master/perceptron/src/python) | Perceptron | 2 | 300, 200 (test) | fit_transform/transform (StandardScaler) | no | 0% | Perceptron | train/test sets |  | Accuracy |
| [svm](https://github.com/zazhigin/hse/tree/master/svm/src/python) | SVM | 2 | 10 | no | no | 0% | SVC |  |  |  |
| [svm-texts](https://github.com/zazhigin/hse/tree/master/svm-texts/src/python) | 20 newsgroups | text (28382) | 1786 | fit_transform (TF/IDF) | no | 99.99% | SVC (TF/IDF) | KFold(5) |  | coef |
| [logistic](https://github.com/zazhigin/hse/tree/master/logistic/src/python) | Logistic | 2 | 205 | no | no | 0% | LogisticRegression (own impl.) |  |  | AUC-ROC |
| [metrics](https://github.com/zazhigin/hse/tree/master/metrics/src/python) | 2 datasets |  | 200 | no | no |  |  |  |  | Accuracy, Precision, Recall, F1 |
| [linreg](https://github.com/zazhigin/hse/tree/master/linreg/src/python) | CV/Salary | text (24627) | 60000 | fit_transform/transform (TF/IDF) | no | 99.99% | Ridge (TF/IDF) | train/test sets |  |  |
| [pca](https://github.com/zazhigin/hse/tree/master/pca/src/python) | Close prices | 30 | 374 | no | no | 0% | PCA | close/djia sets | n_components=10 | corrcoef |
| [forest](https://github.com/zazhigin/hse/tree/master/forest/src/python) | Abalone | 8 | 4177 | no | no | 4.02% | RandomForestRegressor | KFold(5) | n_estimators=1/51 | R2 |
| [gbm](https://github.com/zazhigin/hse/tree/master/gbm/src/python) | Bioresponse | 1776 | 3751 | no | no | 83.96% | GradientBoostingClassifier, RandomForestClassifier | train/test=0.8 | n_estimators | log-loss |
| [clustering](https://github.com/zazhigin/hse/tree/master/clustering/src/python) | Parrots (713x474) | 3 (RGB) | 337962 | no | no | 8.35% | KMeans |  | n_clusters=1/21 | PSNR, MSE |
| [dota2](https://github.com/zazhigin/hse/tree/master/dota2/src/python) | Dota 2 | 101 (+108 words bag) | 97230, 17177 (test) | scale | yes | 19.37% (49.49%) | GradientBoostingClassifier, LogisticRegression | KFold(5) | n_estimators, C | AUC-ROC |
