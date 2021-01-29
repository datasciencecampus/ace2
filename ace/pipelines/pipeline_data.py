"""Sections of this code are based on scikit-learn sources; scikit-learn code is covered by the following license:
New BSD License
Copyright (c) 2007–2018 The scikit-learn developers.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
import bz2
import json
import os
import pickle
import string
import re

import joblib

import numpy as np
import pandas as pd

from os import path
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ace.utils.utils import create_load_balance_hist, check_and_create


def configure_pipeline(experiment_path, drop_nans=True, load_balance_ratio=None, keep_headers=['RECDESC'],
                       label_column='CHECKTHIS', plot_classes=False, drop_classes_less_than=0,
                       drop_classes_more_than=None):

    base_path = path.join(experiment_path, 'data')
    config_path = path.join(base_path, 'config.json')

    d = {
        'drop_nans': drop_nans,
        'load_balance_ratio': load_balance_ratio,
        'drop_classes_less_than': drop_classes_less_than,
        'drop_classes_more_than': drop_classes_more_than,
        'keep_headers': keep_headers,
        'label_column': label_column,
        'plot_classes': plot_classes,
        'base_path': base_path
    }
    check_and_create(base_path)
    with open(config_path, mode='w+') as fp:
        json.dump(d, fp, indent=4)


class DropNans(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return X.dropna()


class DropClasses(BaseEstimator, TransformerMixin):
    def __init__(self, label_col, minimum=0, maximum=np.inf):
        self.__min = minimum
        self.__max = maximum
        self.__label_col = label_col

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):

        # Find the labels we wish to keep
        counts = Counter(X[self.__label_col])
        keep_labels = [i for i in counts.keys() if (counts[i] >= self.__min) & (counts[i] <= self.__max)]

        mask = np.array([True if(label in keep_labels) else False for label in X[self.__label_col]])

        return X[mask]


class LoadBalance(BaseEstimator, TransformerMixin):
    def __init__(self, ratio=1.0):
        self.__ratio = ratio

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return X


class PlotData(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        print("Not removing stopwords")

        return X


class KeepHeaders(BaseEstimator, TransformerMixin):

    def __init__(self, headers):
        self.__headers = headers

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return X[self.__headers]


class PipelineData:
    def __init__(self, experiment_path):

        base_path = path.join(experiment_path, 'data')
        config_path = path.join(base_path, 'config.json')
        global config_test

        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        self.__base_path = base_path
        self.__pipeline_steps = []
        self.__pipe = None
        config_test = self.__config

    def extend_pipe(self, steps):

        self.__pipeline_steps.extend(steps)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, y=None):
        """
        Combines preprocessing steps into a pipeline object
        """

        # Empty old pipeline to avoid interesting errors after repeated calls
        self.__pipeline_steps = []

        if self.__config['keep_headers']:
            self.__pipeline_steps.append(('keep_headers',
                                   KeepHeaders(headers=self.__config['keep_headers'] +
                                                       [self.__config['label_column']])))

        if self.__config['drop_nans']:
            self.__pipeline_steps.append(('drop_nans', DropNans()))

        if self.__config['drop_classes_less_than']:
            self.__pipeline_steps.append(('drop_classes_less_than',
                                   DropClasses(label_col=self.__config['label_column'],
                                               minimum=self.__config['drop_classes_less_than'])))

        if self.__config['drop_classes_more_than']:
            self.__pipeline_steps.append(('drop_classes_more_than',
                                   DropClasses(label_col=self.__config['label_column'],
                                               maximum=self.__config['drop_classes_more_than'])))

        if self.__config['load_balance_ratio']:
            self.__pipeline_steps.append(('load_balance_ratio', LoadBalance(ratio=self.__config['load_balance_ratio'])))

        if self.__config['plot_classes']:
            self.__pipeline_steps.append(('plot_classes', PlotData()))

        self.__pipe = Pipeline(self.__pipeline_steps)

        print("Fitting data pipeline")
        self.__pipe.fit(X)

    def transform(self, X, y=None):
        print("Transforming data pipeline")
        X = self.__pipe.transform(X)
        return X.drop(self.__config['label_column'], axis=1), list(X[self.__config['label_column']])

    def split_fit_transform_save(self, X, outfile_name='processed.pkl.bz2', split_valid=None):

        if split_valid:
            msk = np.random.rand(len(X)) > split_valid
            train_X, train_y = self.fit_transform(X[msk])
            valid_X, valid_y = self.transform(X[~msk])

            train_obj = train_X, train_y
            valid_obj = valid_X, valid_y

            with bz2.BZ2File(path.join(self.__base_path, "train" + outfile_name), 'wb') as pickle_file:
                pickle.dump(train_obj, pickle_file, protocol=4, fix_imports=False)

            with bz2.BZ2File(path.join(self.__base_path, "valid" + outfile_name), 'wb') as pickle_file:
                pickle.dump(valid_obj, pickle_file, protocol=4, fix_imports=False)

            # Return train_X, train_y, valid_X, valid_y
            return None

        X, y = self.fit_transform(X)

        with bz2.BZ2File(path.join(self.__base_path, outfile_name), 'wb') as pickle_file:
            pkl_obj = X, y
            pickle.dump(pkl_obj, pickle_file, protocol=4, fix_imports=False)

        return None
