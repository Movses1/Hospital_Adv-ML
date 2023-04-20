from preprocessor import Preprocessor
import numpy as np
import pandas as pd
import sklearn

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, BatchNormalization, Reshape, Bidirectional, \
    GaussianNoise, LSTM
from tensorflow.keras import Sequential

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from matplotlib import pyplot as plt

df = pd.read_csv('hospital_deaths_train.csv', index_col='recordid')
transformer = Preprocessor()
# transformer.fit(df).to_csv('transformed data diffs.csv')
y = df['In-hospital_death']
x = transformer.fit(df)
print(x.shape, y.shape)

poly = PolynomialFeatures(degree=2, interaction_only=True)
poly.set_output(transform='pandas')
#x = poly.fit_transform(x)
"""x.drop(['NISysABP_fl_isna',  'NIMAP_fl_isna',
        'DiasABP_fl_isna', 'MAP_fl_isna',
        'PaO2_fl_isna', 'pH_fl_isna',
        'BUN_fl_isna',
        'HR_fl_isna', 'GCS_fl_isna',
        'ALT_fl_isna', 'AST_fl_isna'], axis=1, inplace=True)"""
print(x.shape)

def build_model():
    model = Sequential()
    model.add(Input(shape=(x.shape[1],)))
    model.add(Dense(2048, activation='relu'))
    model.add(GaussianNoise(0.1, seed=1))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC()],
    )
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=1)
train_scores, train_cms, scores, cms = [], [], [], []

for train_index, test_index in kf.split(x):
    clf = build_model()
    clf.fit(x.iloc[train_index], y.iloc[train_index],
            batch_size=32,
            epochs=20,
            class_weight={0: 1, 1: 6.2}, verbose=2
            )

    train_probs = clf.predict(x.iloc[train_index])
    train_preds = train_probs >= 0.5
    probs = clf.predict(x.iloc[test_index])
    preds = probs >= 0.5

    train_acc = accuracy_score(y.iloc[train_index], train_preds)
    train_auc = roc_auc_score(y.iloc[train_index], train_probs)
    train_cm = confusion_matrix(y.iloc[train_index], train_preds)

    acc = accuracy_score(y.iloc[test_index], preds)
    auc = roc_auc_score(y.iloc[test_index], probs)
    cm = confusion_matrix(y.iloc[test_index], preds)

    print(acc, auc)
    print(cm)
    train_scores.append(np.array([train_acc, train_auc]))
    train_cms.append(train_cm)
    scores.append(np.array([acc, auc]))
    cms.append(cm)
    roc = roc_curve(y.iloc[test_index], probs)
    plt.plot(roc[0], roc[1])

print()
print(np.array(train_scores).mean(axis=0))
print(np.array(train_cms).mean(axis=0), '\n')
print(np.array(scores).mean(axis=0))
print(np.array(cms).mean(axis=0))

plt.show()

# scores = cross_val_score(clf, x, y, cv=5, scoring='roc_auc')
# print(scores)
# print(scores.mean(axis=0))
