# this should include class named Model, which should include fit and predict functions.
# It should be considered, that the input is already preprocessed here.

from preprocessor import Preprocessor
import numpy as np
import pandas as pd
import sklearn
import pickle

from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Model:
    def __init__(self):
        self.__fitted = False
        stacker_m = Model.__stacker_m()
        stacker_a = Model.__stacker_a()
        estimators = [
            ('stacker_m', stacker_m),
            ('stacker_a', stacker_a)
        ]
        self.model = StackingClassifier(estimators=estimators,
                                        final_estimator=LogisticRegression(penalty='l2', class_weight='balanced',
                                                                           max_iter=int(1e5), random_state=1),
                                        stack_method='predict_proba',
                                        verbose=3,
                                        n_jobs=2)

    def fit(self, x, y):
        self.model.fit(x, y)
        self.__fitted = True
        print('Fitted\nsaving model...')

        with open('final_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print('model saved')

    def predict(self, x):
        if not self.__fitted:
            with open('final_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        return self.model.predict(x)

    @staticmethod
    def __stacker_m():
        clf_m_1 = XGBClassifier(learning_rate=0.1,
                                max_depth=10,
                                max_leaves=20,
                                alpha=0.1,
                                scale_pos_weight=3,
                                num_parallel_tree=100,
                                random_state=1)
        clf_m_2 = SVC(class_weight='balanced', probability=True, random_state=1)
        clf_m_3 = GaussianNB()
        clf_m_4 = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini',
                                                                     max_depth=10,
                                                                     max_leaf_nodes=20,
                                                                     splitter='best',
                                                                     class_weight=None),
                                    n_estimators=100, n_jobs=5, random_state=1)
        estimators_m = [
            ('SVC', clf_m_2),
            ('GNB', clf_m_3),
            ('BC', clf_m_4),
        ]
        stacker_m = StackingClassifier(estimators=estimators_m,
                                       final_estimator=clf_m_1,
                                       passthrough=True,
                                       stack_method='predict_proba',
                                       verbose=3,
                                       n_jobs=2)
        return stacker_m

    @staticmethod
    def __stacker_a():
        clf_a_1 = KNeighborsClassifier(n_neighbors=25, weights="distance")
        clf_a_2 = LinearDiscriminantAnalysis()
        clf_a_3 = BaggingClassifier(estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10,
                                                                                                  max_leaf_nodes=20,
                                                                                                  class_weight={0: 1,
                                                                                                                1: 1.3},
                                                                                                  random_state=1),
                                                                 n_estimators=35,
                                                                 learning_rate=0.3, random_state=1),
                                    n_estimators=37, random_state=1)
        estimators = [
            ('KNN', clf_a_1),
            ('LDA', clf_a_2),
        ]
        stacker_a = StackingClassifier(estimators=estimators,
                                       final_estimator=clf_a_3,
                                       passthrough=True,
                                       stack_method='predict_proba',
                                       verbose=3,
                                       n_jobs=2)
        return stacker_a
