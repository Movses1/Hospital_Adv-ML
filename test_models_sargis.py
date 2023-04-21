import numpy as np
import pandas as pd
import Preprocessor
import argparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import itertools
import joblib
import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help="data_path the int you use here")

parser.add_argument('--model', type=str, help="data_path the int you use here")

args = parser.parse_args()
args_RF = parser.parse_args(['--model', 'best_model_rand.pkl'])
args_QDA = parser.parse_args(['--model', 'best_model_QDA.pkl'])
args_LR = parser.parse_args(['--model', 'best_model_LogReg.pkl'])

data_path = args.data_path
model_path_RF = args_RF.model
model_path_QDA = args_QDA.model
model_path_LR = args_LR.model

new_data = pd.read_csv(data_path)

print(new_data.shape)

X = np.array(new_data.drop(['In-hospital_death'], axis=1))
y = np.array(new_data['In-hospital_death'])


print(X.shape)

model_RF = joblib.load(model_path_RF)
model_QDA = joblib.load(model_path_QDA)
model_LR = joblib.load(model_path_LR)

y_pred_QDA = model_QDA.predict(X)
y_pred_LR = model_LR.predict(X)
y_pred_RF = model_RF.predict(X)

threshold = 0.5
y_pred_binary_rf = [1 if pred >= threshold else 0 for pred in y_pred_RF]
y_pred_binary_qda = [1 if pred >= threshold else 0 for pred in y_pred_QDA]
y_pred_binary_lr = [1 if pred >= threshold else 0 for pred in y_pred_LR]

print("roc_auc_score RF", roc_auc_score(y, y_pred_binary_rf))
print("Accuracy _rf:", accuracy_score(y, y_pred_binary_rf))

print("roc_auc_score QDA", roc_auc_score(y, y_pred_binary_qda))
print("Accuracy QDA:", accuracy_score(y, y_pred_binary_qda))

print("roc_auc_score QDA", roc_auc_score(y, y_pred_binary_lr))
print("Accuracy QDA:", accuracy_score(y, y_pred_binary_lr))