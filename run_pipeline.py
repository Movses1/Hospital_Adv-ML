# this should include a class named Pipeline, which should have a run method.
# This file can have 2 arguments. First is data path (--data_path), absolute path to a train or
# test dataset. Second is whether to run model in a training mode or in testing mode (--inference),
# if the argument is not given, it is training mode, otherwise - it is testing mode.

import argparse
from preprocessor import Preprocessor
from model import Model
import numpy as np
import pandas as pd
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="hospital_deaths_train.csv", help="path to hospital.csv")
parser.add_argument("--inference", type=bool, default=False, help="run in inference mode or not: default=False")
args = parser.parse_args()


class Pipeline:
    def __init__(self):
        self.df = pd.read_csv(args.data_path, index_col='recordid')
        self.inference = args.inference
        self.model = Model()
        self.transformer = Preprocessor()

    def run(self, test=args.inference):
        """
        runs the pipeline fitting and saving the model
        or running it on test data
        all the outputs will be saved in files
        :return: None
        """
        if test:
            x = self.transformer.transform(self.df)
            preds = self.model.predict(x)
            dump_dict = {'predict_probas': preds.tolist(), 'threshold': 0.42}
            with open('predictions.json', 'w') as f:
                json.dump(dump_dict, f)

        else:
            # loading Model from our model class
            self.model = Model()
            X, Y = self.df.drop('In-hospital_death', axis=1), self.df['In-hospital_death']
            X = self.transformer.fit_transform(self.df)
            self.model.fit(X, Y)

obj = Pipeline()
obj.run()