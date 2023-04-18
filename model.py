# this should include class named Model, which should include fit and predict functions.
# It should be considered, that the input is already preprocessed here.

from preprocessor import Preprocessor as prpr
import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

class Model:
    def __init__(self):
        estimators = [
            ('RF', RandomForestClassifier(n_estimators=100, random_state=1)),
            ('ADB', AdaBoostClassifier(n_estimators=100, random_state=1)),
            ('bag_clf', BaggingClassifier(n_estimators=100, random_state=1)),
            ('grad_b', GradientBoostingClassifier())
        ]
        self.model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())

    def fit(self, X, Y):
        self.model.fit(X, Y) # sample_weights = ..., shape: n_samples
        return self.model.score(X, Y)

    def predict(self, X):
        return self.model.predict(X)

