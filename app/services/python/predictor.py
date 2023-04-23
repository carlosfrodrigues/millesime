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
            'density',
            'pH',
            'sulphates',
            'free sulfur dioxide',
            'alcohol',
            'total sulfur dioxide']
    values = dict(zip(keys, parameters))
    df = pd.DataFrame(values, index=[0])
    prediction = clf.predict(df)
    return prediction[0]