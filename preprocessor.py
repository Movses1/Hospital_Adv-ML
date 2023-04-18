# this file should include a class named Preprocessor, which should have fit and transform methods.
# All the preprocessing stages should be done here - filling nans, scaling, feature extraction

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self):
        self.df_orig = None  # train dataframe
        self.orig_means = None  # mean values for each column in train data
        self.orig_medians = None  # median values for each column in train data
        self.scaler = MinMaxScaler()  # Warning: scaler should be fitted only on train data
        self.df_orig_tr = None  # original df but transformed
        self.gender_heights = np.array([160.5, 176])

        self.unique_tests = []  # unique medical test names that have extensions
        self.unique_tests_singular = []  # unique medical tests that don't have any extensions

    def fit(self, df_new):
        """
        :param df_new:
        fits the scaler after performing the transformations
        and saves means and medians of the original data
        :return: df_new but transformed
        """
        self.df_orig = df_new
        if 'In-hospital_death' in df.columns:
            self.df_orig.drop('In-hospital_death', axis=1, inplace=True)
        self.orig_means = self.df_orig.mean(axis=0)
        self.orig_medians = self.df_orig.median(axis=0)
        self.df_orig_tr = df.copy()

        # creating _isna columns for tests with extensions
        test_name = np.array([i[:i.find('_')] for i in df_new.columns if '_' in i])
        test_name_singular = np.array([i for i in df_new.columns if '_' not in i])
        self.unique_tests = np.unique(test_name)
        self.unique_tests_singular = np.unique(test_name_singular)
        has_nans = df_new[self.unique_tests_singular].isna().any()
        self.unique_tests_singular = self.unique_tests_singular[has_nans]

        self.df_orig_tr = self.transform(self.df_orig)
        self.df_orig_tr = self.scaler.fit_transform(self.df_orig_tr)
        return self.df_orig_tr

    def transform(self, df_new):
        """
        :param df_new, save_isnas:
        save_isnas - weather or not to save the _isna column names
        transforms the original dataframe by adding new columns and handling nan values
        :return: df_transformed
        """
        df_new.Gender.fillna(1, inplace=True)
        df_transformed = df_new.copy()
        height_nans = df_transformed.Height.isna()
        df_transformed[height_nans].Height = self.gender_heights[df_transformed[height_nans].Gender.values.astype('int32')]

        df_transformed[self.unique_tests + '_fl_isna'] = df_new[self.unique_tests + '_first'].isna()
        df_transformed[self.unique_tests_singular] = df_new[self.unique_tests_singular].isna()

        # you choose how you fill the nans
        df_transformed.fillna(self.orig_means, inplace=True)
        return df_transformed


df = pd.read_csv('hospital_deaths_train.csv')
