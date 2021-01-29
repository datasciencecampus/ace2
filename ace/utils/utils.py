import os
import pickle
import datetime

from os import path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def unix_time_secs(dt):
    """ NOT CURRENTLY USED BY ANYTHING. """
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds()


def create_load_balance_hist(yin):
    d = {}
    y = yin.tolist()
    for i in range(len(yin)):
        key = y[i]
        if key in d:
            d[key] += 1
        else:
            d[key] = 1
    return d


def strip_file_ext(file_path, file_ext='.csv'):
    return file_path.replace(file_ext, '')


def check_and_create(base_path):
    if not path.exists(base_path):
        os.makedirs(base_path)


def load_corrections(parser_dir, dict_files_list=None):
    """
    Loads a series of dictionaries of manually defined regex replacements/alterations to be made to the text.
    :param parser_dir: path to folder with the pickled dicts in
    :param dict_files_list: list of filenames, if None all files in folder will be read
    :return a single bigger dict of all corrections
    """
    def __load(path, filename):
        try:
            with open(f"{path}/{filename}", 'rb') as handle:
                return pickle.load(handle)
        except IsADirectoryError:
            pass

    def __load_dicts(folder, filenames):
        try:
            return {
                filename.split('.')[0]: __load(folder, filename)
                for filename in filenames
            }
        except FileNotFoundError:
            raise FileNotFoundError(f"parser files not found in {path}")

    if not dict_files_list:
        dict_files_list = os.listdir(parser_dir)

    return __load_dicts(parser_dir, dict_files_list)


class DataBalancer(BaseEstimator, TransformerMixin):
    """
    Handles rebalancing of dataset through subsampling
    """
    def __init__(self,
                 min_class_support=20,
                 max_class_support=None,
                 balanced=False,
                 random_state=42):

        self._class_list = None
        self._min_class_support = min_class_support
        self._max_class_support = max_class_support
        self._balanced = balanced
        self._random_state = random_state

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X=None, y=None):
        # Determine number from each class to be sampled
        dynamic_limit = int(len(y) / len(pd.unique(y)))
        self._min_class_support = np.max([dynamic_limit, self._min_class_support])

        # Identify the classes that are populous enough in the fitted data to be included
        class_counts = pd.Series(y).value_counts()
        self._class_list = list(class_counts[class_counts > (2 * self._min_class_support)].index)

        return self

    def transform(self, X, y):
        # If it's already a series, nothing will change.  This ensures there's an index
        # to use for sample selection
        ys = pd.Series(y)

        # Filter to classes that have 2 x minimum required support
        ys = ys[ys.isin(self._class_list)]

        # Get the indices of (randomly selected) sub-sample
        selection_index = ys.groupby(ys) \
            .sample(self._min_class_support, random_state=self._random_state) \
            .index

        # Convert to boolean mask so it can be used with other data
        sample_mask = pd.Series(y).index.isin(selection_index)

        return X[sample_mask], y[sample_mask]
