"""
 - 15/10/2020

This needs to take the ml classification results of two models, for the same sets of samples, and report how the two
systems compare to one-another.

Do we also want to put single system QA/metrics stuff in here later?

"""

import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


def fake_test_data(sample_size=20, stringy=False):
    """
    Temporary, for getting this module up and running
    :param sample_size: Number of samples to fake
    :param stringy: Whether it should be string or int
    :return: y1, y2, y - three 1D arrays of size sample_size
    """
    import random

    def one_set(sample_size, stringy):
        return np.asarray(["C" + str(random.randint(0, 3)) if(stringy)
                           else random.randint(0, 5)
                           for x in range(sample_size)])

    y1 = one_set(sample_size, stringy)
    y2 = one_set(sample_size, stringy)
    y = one_set(sample_size, stringy)

    return y1, y2, y


def check_types_match():
    """
    Test that the two system's guesses, and the 'true' values, are all of the same data type
    :return:
    """
    return 0


def right_pad():
    """
    Utility to right pad with zeros str-types, poss remove, not needed outside of SIC SOC project
    :return:
    """


def join_results(y1, y2, y, on_index=False):
    """
    Take three iterables (pandas series, array, list), join to create a pandas DataFrame.
    If on_index, join using index values (assumed to indicate source Sample).
    :param y1: predicted classes for series of samples from system 1
    :param y2: predicted classes for series of samples from system 2
    :param y: true classes of series of samples
    :param on_index: bool, whether to join using series indexes or just assume ordered
    :return: pandas.DataFrame containing columns for each input series
    """

    if on_index:
        df = pd.concat([y1, y2, y], axis=1)
        df.columns = ["y1", "y2", "y"]
    else:
        df = pd.DataFrame({"y1": y1,
                           "y2": y2,
                           "y": y})
    return df


def compare_systems_by_class(y1, y2, y, on_index=False):
    """
    Compares by-class performance of two ML systems, including their overlap
    :param y1: predicted classes for series of samples from system 1
    :param y2: predicted classes for series of samples from system 2
    :param y: true classes of series of samples
    :return: pandas.DataFrame containing results of the comparison
    """
    df = join_results(y1, y2, y, on_index)

    # All the bool!  These by-Sample comparisons form the base of the various aggregations
    df['y1_correct'] = df['y1'] == df['y']
    df['y2_correct'] = df['y2'] == df['y']
    df['both_correct'] = df['y1_correct'] & df['y2_correct']
    df['either_correct'] = df['y1_correct'] | df['y2_correct']
    df['one_correct'] = (df['either_correct'] == True) & (df['both_correct'] == False)

    # Aggregate by class
    df['support'] = 1
    overlap_df = df.groupby("y") \
        .agg({"y1_correct": "sum",
              "y2_correct": "sum",
              "both_correct": "sum",
              "either_correct": "sum",
              "one_correct": "sum",
              "support": "count"}) \
        .reset_index()

    overlap_df['overlap'] = 100.0 * overlap_df['both_correct'] / overlap_df['either_correct']

    # Calculate by-class recall for different systems/combinations
    for col in overlap_df.columns:
        if "correct" in col:
            overlap_df[str(col).replace("correct", "recall")] = 100.0 * overlap_df[col] / overlap_df['support']

    return overlap_df


# Temp scripting while I get this show on the road
y1, y2, y = fake_test_data(100, stringy=True)

df = compare_systems_by_class(y1, y2, y)

print(df)
print(df[['y1_correct', 'y1_recall', 'support']])
print(df[['both_correct', 'both_recall', 'support']])


# from os import path
# from sklearn import metrics as met
# from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, balanced_accuracy_score, zero_one_loss, f1_score, confusion_matrix, roc_curve, auc
#
# from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.preprocessing import label_binarize, normalize
#
# from scripts.qa_results import qa_results_by_class, join_compare_systems
#
# def configure_pipeline(experiment_path, data_path, alg_type, multi=True, dirname='soc',
#                   data_filename='training_data.pkl.bz2',
#                   train_test_ratio=0.75, threshold=0.5, accuracy=0.9):
#     base_path = path.join(experiment_path, 'features')
#     config_path = path.join(base_path,  'config.json')
#     pipe_path = path.join(base_path, 'pipe')
#     d={
#
#         'alg_type':alg_type,
#         'multi': multi,
#         'dirname': dirname,
#         'data_filename':data_filename,
#         'train_test_ratio':train_test_ratio,
#         'threshold': threshold,
#         'accuracy':accuracy,
#
#
#     }
#     with open(config_path, 'w') as fp: json.dump(d, fp)
#
#
# class PipelineCompare:
#     def __init__(self, config_filename):
#
#         with open(config_filename, 'w') as fp:
#             self.__config = json.load(fp)
#
#
#     def fit(self, X, y):
#         print()
#
#
#     def transform(self, X, y):
#         print()
