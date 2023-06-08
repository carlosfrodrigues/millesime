import joblib
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

 
dataset= open('winequality-red.csv')
data = pd.read_csv(dataset, sep=';')

y = data.quality
X = data.drop('quality', axis='columns')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)

if len(sys.argv) > 1:
    if sys.argv[1] == 'adaboost_extra': #R2 Score: 0.4721230098674255  MSE: 0.340625
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                            AdaBoostRegressor(estimator=ExtraTreeRegressor(), random_state=123))
        parameters = {'adaboostregressor__learning_rate': [0.01, 0.1, 0.5, 1.0],
                      'adaboostregressor__loss': ['linear', 'square', 'exponential'],
                      'adaboostregressor__n_estimators': [50, 100, 200]}
    elif sys.argv[1] == 'adaboost': #R2 Score: 0.29882900753251374  MSE: 0.45244701658478154
            pipeline = make_pipeline(preprocessing.StandardScaler(),
                                    AdaBoostRegressor(random_state=123))
            parameters = {'adaboostregressor__learning_rate': [0.01, 0.1, 0.5, 1.0],
                          'adaboostregressor__loss': ['linear', 'square', 'exponential'],
                          'adaboostregressor__n_estimators': [50, 100, 200]}
    elif sys.argv[1] == 'bagging_extra': #R2 Score: 0.4776303650342032  MSE: 0.33707125
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                            BaggingRegressor(estimator=ExtraTreeRegressor(), random_state=123))
        parameters = {'baggingregressor__n_estimators': [50, 100, 200],
                      'baggingregressor__max_samples': [0.25, 0.5, 1.0],
                      'baggingregressor__max_features': [0.25, 0.5, 1.0],
                      'baggingregressor__estimator__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                      'baggingregressor__estimator__max_depth': [None, 5, 3, 1],
                      'baggingregressor__estimator__max_features': [0.25, 0.5, 1.0]}
    elif sys.argv[1] == 'bagging': #R2 Score: 0.45518578606453175  MSE: 0.35155414062500007
            pipeline = make_pipeline(preprocessing.StandardScaler(),
                                BaggingRegressor(random_state=123))
            parameters = {'baggingregressor__n_estimators': [50, 100, 200],
                          'baggingregressor__max_samples': [0.25, 0.5, 1.0],
                          'baggingregressor__max_features': [0.25, 0.5, 1.0]}
    elif sys.argv[1] == 'gradient': #R2 Score: 0.4724801749607893  MSE: 0.3403945308524501
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                                GradientBoostingRegressor(random_state=123))
        parameters = {'gradientboostingregressor__max_depth': [None, 5, 3, 1],
                      'gradientboostingregressor__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                      'gradientboostingregressor__learning_rate': [0.01, 0.1, 0.5, 1.0],
                      'gradientboostingregressor__n_estimators': [50, 100, 200],
                      'gradientboostingregressor__max_features': [0.25, 0.5, 1.0],
                      'gradientboostingregressor__criterion': ['squared_error', 'friedman_mse']}
    elif sys.argv[1] == 'hist_gradient': #R2 Score: 0.39414447842185163  MSE: 0.3909424750370873
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                                HistGradientBoostingRegressor(random_state=123))
        parameters = {'histgradientboostingregressor__max_depth': [None, 5, 3, 1],
                      'histgradientboostingregressor__loss': ['squared_error', 'absolute_error', 'poisson', 'quantile'],
                      'histgradientboostingregressor__learning_rate': [0.01, 0.1, 0.5, 1.0]}
    elif sys.argv[1] == 'random_forest': #R2 Score: 0.47407857618499927  MSE: 0.339363125
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                                RandomForestRegressor(random_state=123))
        parameters = {'randomforestregressor__max_depth': [None, 5, 3, 1],
                      'randomforestregressor__n_estimators': [50, 100, 200],
                      'randomforestregressor__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                      'randomforestregressor__max_samples': [0.25, 0.5, 1.0],
                      'randomforestregressor__max_features': [0.25, 0.5, 1.0],}
    elif sys.argv[1] == 'extra_trees': #R2 Score: 0.4898364307766815  MSE: 0.329195
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                            ExtraTreesRegressor(random_state=123))
        parameters = {'extratreesregressor__max_depth': [None, 5, 3, 1],
                      'extratreesregressor__n_estimators': [50, 100, 200],
                      'extratreesregressor__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                      'extratreesregressor__max_features': [0.25, 0.5, 1.0]}
    elif sys.argv[1] == 'ada_bagging_extra': #R2 Score: 0.4928458139112538  MSE: 0.32725312500000003
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                        AdaBoostRegressor(estimator=BaggingRegressor(
                                              estimator=ExtraTreeRegressor())))
        parameters = {'adaboostregressor__learning_rate': [0.01, 0.1, 0.5, 1.0],
                      'adaboostregressor__loss': ['linear', 'square', 'exponential'],
                      'adaboostregressor__n_estimators': [50, 100, 200],
                      'adaboostregressor__estimator__n_estimators': [50, 100, 200],
                      'adaboostregressor__estimator__max_samples': [0.25, 0.5, 1.0],
                      'adaboostregressor__estimator__max_features': [0.25, 0.5, 1.0],
                      'adaboostregressor__estimator__estimator__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}

    clf = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1)

    clf.fit(X_train, y_train)

    pred = clf.best_estimator_.predict(X_test)
    scores = (float(r2_score(y_test, pred)), float(mean_squared_error(y_test, pred)))
    print(clf.best_params_)
    print(f"R2 Score: {scores[0]}  MSE: {scores[1]}")

    joblib.dump(clf.best_estimator_, 'regressor.pkl')

else: #R2 Score: 0.5003836205530543  MSE: 0.32238917859703503
    estimators = [
        ('bagging_extra', BaggingRegressor(random_state=123,
                                        max_features=0.75,
                                        max_samples=1.0,
                                        n_estimators=100,
                                        estimator=ExtraTreeRegressor(
                                            criterion='poisson',
                                            random_state=123))),
        ('ensemble_trees', ExtraTreesRegressor(n_estimators=200,
                                            criterion='squared_error',
                                            max_depth=None,
                                            max_features=0.5,
                                            random_state=123,
                                            n_jobs=-1)),
        ('random_forest', RandomForestRegressor(criterion='poisson',
                                                    random_state=123,
                                                    n_jobs=-1))
    ]
    pipeline = make_pipeline(preprocessing.StandardScaler(),
                                StackingRegressor(estimators=estimators,
                                                  final_estimator=GradientBoostingRegressor(
                                                      random_state=123,
                                                      n_estimators=50,
                                                      learning_rate=0.1,
                                                      criterion='friedman_mse',
                                                      max_features=0.25)))

    pipeline._final_estimator.fit(X_train, y_train)

    pred = pipeline._final_estimator.predict(X_test)
    scores = (float(r2_score(y_test, pred)), float(mean_squared_error(y_test, pred)))
    print(f"R2 Score: {scores[0]}  MSE: {scores[1]}")

    joblib.dump(pipeline._final_estimator, 'regressor.pkl')
