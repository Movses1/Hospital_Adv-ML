# this should include a class named Pipeline, which should have a run method.
# This file can have 2 arguments. First is data path (--data_path), absolute path to a train or
# test dataset. Second is whether to run model in a training mode or in testing mode (--inference),
# if the argument is not given, it is training mode, otherwise - it is testing mode.

import argparse
import pickle
from preprocessor import Preprocessor
from model import Model
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="path to hospital.csv")
parser.add_argument("--inference", type=bool, default=False, help="run in inference mode or not: default=False")
args = parser.parse_args()


class Pipeline:
    def __init__(self):
        self.path = args.data_path
        self.inference = args.inference
        self.model = Model()
        self.transformer = Preprocessor()

    def run(self, test=False):
        """
        runs the pipeline fitting and saving the model
        or running it on test data
        all the outputs will be saved in files
        :return: None
        """
        df = pd.read_csv(self.path)
        if self.inference or test:
            with open('stacking classifier.pkl', 'rb') as file:
                self.model = pickle.load(file)

            X = self.transformer.transform(df)
            preds = pd.Series(self.model.predict(X))
            preds.to_csv('answers.csv')

        else:
            # loading Model from our model class
            self.model = Model()
            X, Y = df.drop('In-hospital_death', axis=1), df['In-hospital_death']
            X = self.transformer.fit(df)
            self.model.fit(X, Y)
            with open('stacking classifier.pkl', 'wb') as file:
                pickle.dump(self.model, file)
