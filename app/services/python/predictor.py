import numpy as np
import pandas as pd
import joblib

def run(parameters):
    clf = joblib.load('app/services/python/regressor.pkl')
    keys = ['fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
	    'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol']
    reorder = [0, 1, 2, 3, 4, 8, 10, 5, 6, 7, 9]
    parameters = [parameters[i] for i in reorder]
    values = dict(zip(keys, parameters))
    df = pd.DataFrame(values, index=[0])
    prediction = clf.predict(df)
    return float(prediction[0])