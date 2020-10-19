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

from sklearn.metrics import classification_report


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


def expanded_right_pad(x, pad_len=4, pad_char="0"):
    # Do the padding
    padded = np.array([str(entry).ljust(pad_len, pad_char) for entry in x])
    # Fix the fail codes
    padded = np.where(padded == "-100", "-1", padded)
    padded = np.where(padded == "-600", "-6", padded)
    return padded


def join_results(y1, y2, y_true, on_index=False):
    """
    Take three iterables (pandas series, array, list), join to create a pandas DataFrame.
    If on_index, join using index values (assumed to indicate source Sample).
    :param y1: predicted classes for series of samples from system 1
    :param y2: predicted classes for series of samples from system 2
    :param y_true: true classes of series of samples
    :param on_index: bool, whether to join using series indexes or just assume ordered
    :return: pandas.DataFrame containing columns for each input series
    """
    # Check the types are the same
    y1_types = [type(x) for x in y1]
    y2_types = [type(x) for x in y2]
    y_true_types = [type(x) for x in y_true]

    if not y1_types == y2_types == y_true_types:
        raise Exception("The types of the predictions and the true values are not all the same!")

    if on_index:
        df = pd.concat([y1, y2, y_true], axis=1)
        df.columns = ["y1", "y2", "y_true"]
    else:
        df = pd.DataFrame({"y1": y1,
                           "y2": y2,
                           "y_true": y_true})
    return df


def by_class_compare(y1, y2, y_true, on_index=False):
    """
    Compares by-class performance of two ML systems, including their overlap
    :param y1: predicted classes for series of samples from system 1
    :param y2: predicted classes for series of samples from system 2
    :param y_true: true classes of series of samples
    :return: pandas.DataFrame containing results of the comparison
    """
    df = join_results(y1, y2, y_true, on_index)

    # All the bool!  These by-Sample comparisons form the base of the various aggregations
    df['y1_correct'] = df['y1'] == df['y_true']
    df['y2_correct'] = df['y2'] == df['y_true']
    df['both_correct'] = df['y1_correct'] & df['y2_correct']
    df['either_correct'] = df['y1_correct'] | df['y2_correct']
    df['one_correct'] = (df['either_correct'] == True) & (df['both_correct'] == False)

    # Pick out those only one system got correct
    df['ONLY_y1_correct'] = (df['y1'] == df['y_true']) & (df['y2'] != df['y_true'])
    df['ONLY_y2_correct'] = (df['y2'] == df['y_true']) & (df['y1'] != df['y_true'])

    # Aggregate by class
    df['support'] = 1
    overlap_df = df.groupby("y_true") \
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


def by_class_results(y_true, y_pred):
    """
    Report classification performance by class.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: pandas DataFrame of performance by class
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    results = pd.DataFrame(report).transpose().reset_index()
    results.rename(columns={'index': 'class'}, inplace=True)
    results = results[~results.code.isin(['accuracy', 'macro avg', 'weighted avg'])]
    return results


def evaluate_ml_by_class(file_path, matched_only):



# Temp scripting while I get this show on the road
y1, y2, y_true = fake_test_data(100, stringy=True)

df = by_class_compare(y1, y2, y_true)

print(df)

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
def configure_pipeline(experiment_path, data_path, alg_type, multi=True, dirname='soc'):
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

        self._ml_file_path = self.__config('ml_file_path')

        # If the pad options are specified, then do so.
        self._pad_char = self.__config.get("pad_char", None)
        self._pad_len = self.__config.get("pad_len", None)

    def validate_class_stats(self, ml_df=None, matched_only=False):
        """
        Creates a classification report showing the results per class of the ml coding tool
        """

        # If no df provided, assume we're loading one saved elsewhere
        if not ml_df:
            ml_df = pd.read_csv(self._ml_file_path, index_col=0)

        # If only examining those that passed a threshold limit, drop the rest
        if matched_only:
            ml_df = ml_df[ml_df['matched'] == 1]

        if self._pad_char & self._pad_len:
            y_true = expanded_right_pad(ml_df['true_label'], pad_char=self._pad_char, pad_len=self._pad_len)
            y_pred = expanded_right_pad(ml_df['prediction_labels'], pad_char=self._pad_char, pad_len=self._pad_len)

        else:
            y_true = ml_df['true_labels']
            y_pred = ml_df['prediction_labels']

        return by_class_results(y_true, y_pred)