import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
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
 
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100,
                                               random_state=123))
 
parameters = { 'randomforestregressor__max_features' : [1.0, 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
 
clf = GridSearchCV(pipeline, parameters, cv=10)
 
clf.fit(X_train, y_train)
 
pred = clf.predict(X_test)
print( r2_score(y_test, pred) )
print( mean_squared_error(y_test, pred) )
 
joblib.dump(clf, 'regressor.pkl')