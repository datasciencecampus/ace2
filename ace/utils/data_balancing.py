import os
import numpy as np


class DataBalancer:
    """
    Handles selection of specific class subsets and/or rebalancing of asymmetric datasets.
    """
    def __init__(self,
                 class_list=None,
                 min_class_support=20,
                 max_class_support=None,
                 balanced=False):

        self._class_list = class_list
        self._min_class_support = min_class_support
        self._max_class_support = max_class_support
        self._balanced = balanced

    def subset_to_classes(self, X, y):
        # TODO write test that ensures this works with y as pandas Series, np array or list,
        # TODO and with X as pandas Series or DataFrame, ND np array or list
        mask = [entry in self._class_list for entry in y]
        return X[mask], y[mask]

    def fit(self, X, y):
        return None

    def transform(self, X, y):
        return None