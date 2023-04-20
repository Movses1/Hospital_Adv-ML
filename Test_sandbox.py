from preprocessor import Preprocessor
import numpy as np
import pandas as pd
import sklearn

from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

df = pd.read_csv('hospital_deaths_train.csv', index_col='recordid')
transformer = Preprocessor()
# transformer.fit(df).to_csv('transformed data mins.csv')
y = df['In-hospital_death']
x = transformer.fit(df)
print(x.shape, y.shape)

poly = PolynomialFeatures(degree=2, interaction_only=True)
poly.set_output(transform='pandas')
# x = poly.fit_transform(x)
print(x.shape)

clf = XGBClassifier(learning_rate=0.05,
                    max_depth=200,
                    max_leaves=400,
                    alpha=0.1,
                    scale_pos_weight=5,
                    num_parallel_tree=6)
#clf = SVC(probability=True)
#clf = GaussianNB()
'''clf = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini',
                                                         max_depth=200,
                                                         max_leaf_nodes=1000,
                                                         splitter='best',
                                                         class_weight=None),
                        n_estimators=100,
                        n_jobs=-1,
                        random_state=1)'''

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
cms = []
for train_index, test_index in kf.split(x):
    clf.fit(x.iloc[train_index], y.iloc[train_index])
    probs = clf.predict_proba(x.iloc[test_index])

    acc = accuracy_score(y.iloc[test_index], probs[:, 1] > 0.5)
    auc = roc_auc_score(y.iloc[test_index], probs[:, 1])
    cm = confusion_matrix(y.iloc[test_index], probs[:, 1] > 0.5)

    print(acc, auc)
    print(cm)
    scores.append(np.array([acc, auc]))
    cms.append(np.array(cm))

scores = np.array(scores)
cms = np.array(cms)
print()
print(scores.mean(axis=0))
print(cms.mean(axis=0))

# scores = cross_val_score(clf, x, y, cv=5, scoring='roc_auc')
# print(scores)
# print(scores.mean(axis=0))
