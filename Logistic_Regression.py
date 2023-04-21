import numpy as np
import pandas as pd
import Preprocessor
import argparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=argparse.FileType('r'), help="data_path the int you use here")


args = parser.parse_args(['--data_path', 'new_data.csv'])
data_path = args.data_path
print(data_path)

new_data = pd.read_csv(data_path)

X = np.array(new_data.drop(['In-hospital_death'], axis=1))
y = np.array(new_data['In-hospital_death'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


lr = LogisticRegression()

param_grid = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10]}

grid_search = GridSearchCV(lr, param_grid, cv=5)

best_model = grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)

print(len(y_pred), len(y_test))

threshold = 0.5
y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]

print("roc_auc_score", roc_auc_score(y_test, y_pred_binary))
print("Accuracy:", accuracy_score(y_test, y_pred_binary))

joblib.dump(best_model, 'best_model_LogReg.pkl')

