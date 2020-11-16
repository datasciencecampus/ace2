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
import wordninja

from collections import Counter
from datetime import datetime
from os import path

from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker
from sklearn.pipeline import Pipeline

import pandas as pd

from ace.utils.utils import create_load_balance_hist, check_and_create


def configure_pipeline(experiment_path, data_path, drop_nans=True, load_balance_ratio=0.0, keep_headers=['RECDESC'],
                       plot_classes=True, drop_classes_less_than=0, drop_classes_more_than=0):
    base_path = path.join(experiment_path, 'data')
    config_path = path.join(base_path, 'config.json')
    d = {
        'drop_nans': drop_nans,
        'load_balance_ratio': load_balance_ratio,
        'drop_classes_less_than': drop_classes_less_than,
        'drop_classes_more_than': drop_classes_more_than,
        'keep_headers': keep_headers,
        'plot_classes': plot_classes,
        'data_path': data_path,
        'base_path': base_path
    }
    check_and_create(base_path)
    with open(config_path, mode='w+') as fp:
        json.dump(d, fp)



class DropNans(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return self

class DropClasses(BaseEstimator, TransformerMixin):
    def __init__(self, minimum=0, maximum=0):
        self.__min=minimum
        self.__max=maximum


    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return self


class LoadBalance(BaseEstimator, TransformerMixin):
    def __init__(self, ratio=0.0):
        self.__ratio

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return self


class KeepHeaders(BaseEstimator, TransformerMixin):
    def __init__(self, headers=[]):
        self.__headers=headers

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return self


class PlotData(BaseEstimator, TransformerMixin):


    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        print("removing stopwords")

        return self



class PipelineData:
    def __init__(self, experiment_path):

        base_path = path.join(experiment_path, 'data')
        config_path = path.join(base_path, 'config.json')
        global config_test
        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        self.__pipeline_steps = []
        self.__pipe = None
        config_test = self.__config

    def extend_pipe(self, steps):

        self.__pipeline_steps.extend(steps)

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X=None, y=None):

        if X is None:
            with bz2.BZ2File(self.__config['data_path'], 'rb') as pickle_file:
                X, y = pickle.load(pickle_file)

        """
        Combines preprocessing steps into a pipeline object
        """

        pipeline_steps = []
        if self.__config['keep_headers']:
            pipeline_steps.append(('keep_headers', KeepHeaders(headers=self.__config['keep_headers'])))
        if self.__config['drop_nans']:
            pipeline_steps.append(('drop_nans', DropNans()))
        if self.__config['drop_classes_less_than']:
            pipeline_steps.append(('drop_classes_less_than', DropClasses(minimum=self.__config['drop_classes_less_than'])))
        if self.__config['drop_classes_more_than']:
            pipeline_steps.append(('drop_classes_more_than', DropClasses(minimum=self.__config['drop_classes_more_than'])))
        if self.__config['load_balance_ratio']:
            pipeline_steps.append(('load_balance_ratio', LoadBalance(ratio=self.__config['load_balance_ratio'])))
        if self.__config['plot_classes']:
            pipeline_steps.append(('plot_classes', PlotData()))

        self.__pipeline_steps.extend(pipeline_steps)

        self.__pipe = Pipeline(self.__pipeline_steps)

        for header in self.__config['text_headers']:
            print("Fitting pipeline for " + header)
            X_i = X[header].astype(str)
            self.__pipe.fit(X_i, y)


    def transform(self, X=None, y=None):

        if X is None:
            with bz2.BZ2File(self.__config['data_path'], 'rb') as pickle_file:
                X, y = pickle.load(pickle_file)


        self.__pipe.transform()

        return X

