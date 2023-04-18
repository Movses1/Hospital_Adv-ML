from preprocessor import Preprocessor as prpr
import numpy as np
import pandas as pd
import sklearn

df = pd.read_csv('hospital_deaths_train.csv', index_col='recordid')
transformer = prpr()
transformer.fit(df).to_csv('transformed data.csv')