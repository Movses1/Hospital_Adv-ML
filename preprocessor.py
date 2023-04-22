# this file should include a class named Preprocessor, which should have fit and transform methods.
# All the preprocessing stages should be done here - filling nans, scaling, feature extraction

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import pickle


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
        self.df_orig_tr = self.df_orig.copy()
        if 'In-hospital_death' in self.df_orig_tr.columns:
            self.df_orig_tr.drop('In-hospital_death', axis=1, inplace=True)
        self.orig_means = self.df_orig_tr.mean(axis=0)
        self.orig_medians = self.df_orig_tr.median(axis=0)
        self.orig_mins = self.df_orig_tr.min(axis=0)

        self.__save_isna_column_names(self.df_orig_tr)

        self.df_orig_tr = self.transform(self.df_orig_tr, fitting=True)
        return self.df_orig_tr

    def transform(self, df_new, fitting=False):
        """
        :param df_new: dataframe we are transforming
        :param fitting: weather or not we want to fit the scaler
        transforms the original dataframe by adding new columns and handling nan values
        :return: df_transformed
        """
        df_transformed = df_new.copy()
        if 'In-hospital_death' in df_transformed.columns:
            df_transformed.drop('In-hospital_death', axis=1, inplace=True)
        gender_nans = df_transformed.Gender.isna()
        np.random.seed(1)
        df_transformed.loc[gender_nans, 'Gender'] = np.random.randint(2, size=gender_nans.sum())

        df_transformed = self.__handle_heights(df_transformed)
        df_transformed = self.__add_isna_columns(df_transformed)

        # you choose how you fill the nans
        df_transformed.fillna(self.orig_means, inplace=True)

        df_transformed = self.__add_bmi(df_transformed)
        df_transformed = self.__add_diff_columns(df_transformed)
        # df_transformed = self.__add_poly_features(df_transformed)

        if fitting:
            self.scaler.fit(df_transformed)
        return self.scaler.transform(df_transformed)

    def __save_isna_column_names(self, df_new):
        # creating _isna columns for tests with extensions
        test_name = np.array([i[:i.find('_')] for i in df_new.columns if '_' in i])
        test_name_singular = np.array([i for i in df_new.columns if '_' not in i])
        self.unique_tests = np.unique(test_name).astype('object')
        self.unique_tests_singular = np.unique(test_name_singular).astype('object')
        has_nans = df_new[self.unique_tests_singular].isna().any()
        self.unique_tests_singular = self.unique_tests_singular[has_nans]

    def __add_isna_columns(self, df_new):
        df_transformed = df_new.copy()
        df_temp = df_new[self.unique_tests + '_first'].isna()
        df_temp.columns = self.unique_tests + '_fl_isna'
        df_transformed = pd.concat([df_transformed, df_temp], axis=1)

        df_temp = df_new[self.unique_tests_singular].isna()
        df_temp.columns = self.unique_tests_singular + '_isna'
        df_transformed = pd.concat([df_transformed, df_temp], axis=1)
        return df_transformed

    def __add_bmi(self, df_new):
        df_transformed = df_new.copy()
        df_transformed.loc[:, 'BMI'] = df_transformed.Weight / df_transformed.Height ** 2
        df_transformed.loc[:, 'BMI_isna'] = df_transformed['Height_isna'] | df_transformed['Weight_isna']
        return df_transformed

    def __add_diff_columns(self, df_new):
        diff = df_new[self.unique_tests + '_last'] - df_new[self.unique_tests + '_first'].values
        diff.columns = self.unique_tests + '_lf_diff'
        df_transformed = pd.concat([df_new, diff], axis=1)

        h_l = np.isin(self.unique_tests + '_highest', df_new.columns)
        diff = df_new[self.unique_tests[h_l] + '_highest'] - df_new[self.unique_tests[h_l] + '_lowest'].values
        diff.columns = self.unique_tests[h_l] + '_hl_diff'
        return pd.concat([df_transformed, diff], axis=1)

    def __handle_heights(self, df_new):
        """
        :param df_new:
        removes outliers, fills nans
        :return:
        """
        df_transformed = df_new.copy()
        df_transformed.loc[df_new.Height > 210, 'Height'] = np.nan
        height_nans = df_transformed.Height.isna()
        df_transformed.loc[height_nans, 'Height'] = self.gender_heights[
            df_transformed[height_nans].Gender.values.astype('int32')]
        df_transformed['Height_isna'] = height_nans

        non_height_cols = self.unique_tests_singular != 'Height'
        self.unique_tests_singular = self.unique_tests_singular[non_height_cols]
        return df_transformed

    def __add_poly_features(self, df_new):
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        poly.set_output(transform='pandas')
        df_transformed = poly.fit_transform(df_new)
        # col_list = []
        # with open('list of column names.txt', 'rb') as f:
        #     col_list = pickle.load(f)
        return df_transformed#[col_list]
