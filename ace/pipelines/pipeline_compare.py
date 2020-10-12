import datetime
import sys
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pdb
import matplotlib.pylab as plt
import os

from os import path, makedirs
from itertools import cycle, product

from sklearn import metrics as met
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, balanced_accuracy_score, zero_one_loss, f1_score, confusion_matrix, roc_curve, auc

from scipy import interpolate
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import label_binarize, normalize

from ace.factories.ml_factory import MLFactory
from ace.utils.utils import create_load_balance_hist, MemoryMonitor
from scripts.qa_results import qa_results_by_class, join_compare_systems

def configure_pipeline(experiment_path, data_path, alg_type, multi=True, dirname='soc',
                  data_filename='training_data.pkl.bz2',
                  train_test_ratio=0.75, threshold=0.5, accuracy=0.9):
    base_path = path.join(experiment_path, 'features')
    config_path = path.join(base_path,  'config.json')
    pipe_path = path.join(base_path, 'pipe')
    d={

        'alg_type':alg_type,
        'multi': multi,
        'dirname': dirname,
        'data_filename':data_filename,
        'train_test_ratio':train_test_ratio,
        'threshold': threshold,
        'accuracy':accuracy,


    }
    with open(config_path, 'w') as fp: json.dump(d, fp)


class PipelineCompare:
    def __init__(self, config_filename):

        with open(config_filename, 'w') as fp:
            self.__config = json.load(fp)


    def fit(self, X, y):
        print()


    def transform(self, X, y):
        print()