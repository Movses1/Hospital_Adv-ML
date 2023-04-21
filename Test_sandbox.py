from preprocessor import Preprocessor
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from custom_metrics import TPR_FPR
from matplotlib import pyplot as plt


df = pd.read_csv('hospital_deaths_train.csv', index_col='recordid')
transformer = Preprocessor()
# transformer.fit(df).to_csv('transformed data diffs.csv')
y = df['In-hospital_death']
x = transformer.fit(df)
print(x.shape, y.shape)

clf0 = LogisticRegression(penalty='l1', max_iter=100000, solver='liblinear', class_weight=None)
clf1 = XGBClassifier(learning_rate=0.1,
                     max_depth=5,
                     max_leaves=10,
                     alpha=0.1,
                     scale_pos_weight=3,
                     num_parallel_tree=100)
clf2 = SVC(class_weight='balanced', probability=True)
clf3 = GaussianNB()
clf4 = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini',
                                                          max_depth=50,
                                                          max_leaf_nodes=200,
                                                          splitter='best',
                                                          class_weight=None),
                         n_estimators=1000, n_jobs=11, random_state=1)
clf5 = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=1)
clf6 = BaggingClassifier(estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10,
                                                                                       max_leaf_nodes=20,
                                                                                       class_weight={0: 1, 1: 1.3}),
                                                      n_estimators=35,
                                                      learning_rate=0.3),
                         n_estimators=37)
estimators = [
    ('XGB', clf1),
    ('SVC', clf2),
    ('GNB', clf3),
    ('BC', clf4),
    # ('ADB', clf5)
]
clf = StackingClassifier(estimators=estimators,
                         # final_estimator=LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga',
                         #                                   class_weight='balanced'),
                         final_estimator=LogisticRegression(penalty='l2', class_weight='balanced'),
                         passthrough=False)

kf = KFold(n_splits=5, shuffle=True, random_state=1)
train_scores, train_cms, scores, cms = [], [], [], []

print(x.shape)
df_tests = 0
df_thresholds = 0
for train_index, test_index in kf.split(x):
    for est in estimators:
        clf = est[1]
        clf.fit(x.iloc[train_index], y.iloc[train_index])
        train_probs = clf.predict_proba(x.iloc[train_index])
        train_preds = clf.predict(x.iloc[train_index])
        preds = clf.predict(x.iloc[test_index])
        probs = clf.predict_proba(x.iloc[test_index])

        train_acc = accuracy_score(y.iloc[train_index], train_preds)
        train_auc = roc_auc_score(y.iloc[train_index], train_probs[:, 1])
        train_cm = confusion_matrix(y.iloc[train_index], train_preds)

        acc = accuracy_score(y.iloc[test_index], preds)
        auc = roc_auc_score(y.iloc[test_index], probs[:, 1])
        cm = confusion_matrix(y.iloc[test_index], preds)

        print(acc, auc)
        print(cm)
        train_scores.append(np.array([train_acc, train_auc]))
        train_cms.append(train_cm)
        scores.append(np.array([acc, auc]))
        cms.append(cm)
        roc = roc_curve(y.iloc[test_index], probs[:, 1])
        plt.plot(roc[0], roc[1], label=est[0])

        df_tst, df_thr = TPR_FPR(roc, est[0], True)
        df_tst.loc[:, 'AUC'] = auc
        if type(df_tests) == int:
            df_tests = df_tst
            df_thresholds = df_thr
        else:
            df_tests = pd.concat([df_tests, df_tst])
            df_thresholds = pd.concat([df_thresholds, df_thr])

        print(clf.classes_)
        print(df_thresholds)
plt.legend()
plt.show()

print('\ntrain scores')
print(np.array(train_scores).mean(axis=0))
print(np.array(train_cms).mean(axis=0), '\ntest scores')
print(np.array(scores).mean(axis=0))
print(np.array(cms).mean(axis=0))
print('\naverage thresholds')
print(df_thresholds.groupby('classifier').mean())
df_thresholds.groupby('classifier').mean().to_csv('thresholds.csv')

sns.heatmap(df_tests.groupby('classifier').mean(), annot=True, cmap="crest")
plt.show()