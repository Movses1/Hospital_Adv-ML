# this file should include a class named Preprocessor, which should have fit and transform methods.
# All the preprocessing stages should be done here - filling nans, scaling, feature extraction

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self):
        self.df_orig = None  # train dataframe
        self.scaler = MinMaxScaler()  # Warning: scaler should be fitted only on train data
        self.scaler.set_output(transform='pandas')
        self.df_orig_tr = None  # original df but transformed

        self.orig_means = None  # mean values for each column in train data
        self.orig_medians = None  # median values for each column in train data
        self.orig_min = None
        self.gender_heights = np.array([160.5, 176])

        self.unique_tests = []  # unique medical test names that have extensions
        self.unique_tests_singular = []  # unique medical tests that don't have any extensions

    def fit(self, df_new):
        """
        :param df_new:
        fits the scaler after performing the transformations
        saves means and medians etc. of the original data
        this functions like a fit_transform basically
        :return: df_new transformed
        """
        self.df_orig = df_new
        if 'In-hospital_death' in self.df_orig.columns:
            self.df_orig.drop('In-hospital_death', axis=1, inplace=True)
        self.orig_means = self.df_orig.mean(axis=0)
        self.orig_medians = self.df_orig.median(axis=0)
        self.orig_mins = self.df_orig.min(axis=0)

        # creating _isna columns for tests with extensions
        test_name = np.array([i[:i.find('_')] for i in df_new.columns if '_' in i])
        test_name_singular = np.array([i for i in df_new.columns if '_' not in i])
        self.unique_tests = np.unique(test_name).astype('object')
        self.unique_tests_singular = np.unique(test_name_singular).astype('object')
        has_nans = df_new[self.unique_tests_singular].isna().any()
        self.unique_tests_singular = self.unique_tests_singular[has_nans]

        self.df_orig_tr = self.transform(self.df_orig, fitting=True)
        return self.df_orig_tr

    def transform(self, df_new, fitting=False):
        """
        :param df_new: dataframe we are transforming
        :param fitting: weather or not we want to fit the scaler
        transforms the original dataframe by adding new columns and handling nan values
        :return: df_transformed
        """
        gender_nans = df_new.Gender.isna()
        df_new.loc[gender_nans, 'Gender'] = np.random.randint(2, size=gender_nans.sum())
        df_transformed = df_new.copy()
        height_nans = df_transformed.Height.isna()
        df_transformed.loc[height_nans, 'Height'] = self.gender_heights[
            df_transformed[height_nans].Gender.values.astype('int32')]
        df_transformed['Height_isna'] = height_nans

        df_temp = df_new[self.unique_tests + '_first'].isna()
        df_temp.columns = self.unique_tests + '_fl_isna'
        df_transformed = pd.concat([df_transformed, df_temp], axis=1)

        df_temp = df_new[self.unique_tests_singular].isna()
        df_temp.columns = self.unique_tests_singular + '_isna'
        df_transformed = pd.concat([df_transformed, df_temp], axis=1)

        # you choose how you fill the nans
        # df_transformed.fillna(self.orig_means, inplace=True)

        if fitting:
            self.scaler.fit(df_transformed)
        return self.scaler.transform(df_transformed)
