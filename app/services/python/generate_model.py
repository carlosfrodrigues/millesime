import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
 
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

y = data.quality
X = data.drop('quality', axis='columns')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
if len(sys.argv) > 1:
    if sys.argv[1] == 'bagging':
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                             BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=100))
        parameters = {'baggingregressor__max_samples': [0.25, 0.5, 1.0],
                        'baggingregressor__max_features': [0.25, 0.5, 1.0]}

else:
    pipeline = make_pipeline(preprocessing.StandardScaler(),
                            RandomForestRegressor(n_estimators=100,
                                               random_state=123))
    parameters = {'randomforestregressor__max_features' : [1.0, 'sqrt', 'log2'],
                              'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1)

clf.fit(X_train, y_train)

pred = clf.best_estimator_.predict(X_test)
print(f"R2 Score: {float(r2_score(y_test, pred))}")
print(f"MSE: {float(mean_squared_error(y_test, pred))}")

joblib.dump(clf.best_estimator_, 'regressor.pkl')