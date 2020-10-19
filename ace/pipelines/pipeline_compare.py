import json
import numpy as np
import pandas as pd

from os import path, makedirs
from sklearn.metrics import classification_report


def configure_pipeline(experiment_path, ml_file_path, comparison_ml_file_path, pad_char, pad_len, matched_only):
    """
    :param experiment_path:
    :param ml_file_path:
    :param comparison_ml_file_path:
    :param pad_char:
    :param pad_len:
    :param matched_only:
    :return:
    """
    base_path = path.join(experiment_path, 'qa')
    config_path = path.join(base_path, 'config.json')

    d = {

        'ml_file_path': ml_file_path,
        'comparison_ml_file_path': comparison_ml_file_path,
        'pad_char': pad_char,
        'pad_len': pad_len,
        'matched_only': matched_only,
        'output_dir': base_path
    }

    if not path.exists(base_path):
        makedirs(base_path)

    with open(config_path, 'w') as fp:
        json.dump(d, fp)


def fake_results(sample_size=20, stringy=False):
    """
    Should've had this in my PhD...

    Create dataframes simulating two Machine Learning system's predictions including whether each sample has been
    filtered ('matched'), according to some probability threshold.

    :param sample_size: Number of samples to fake
    :param stringy: Whether it should be string or int
    :return: df, other_df: pandas DataFrames containing results in same format expected from pipeline_ml
    """
    import random

    def one_set(sample_size, stringy):
        return np.asarray(["C" + str(random.randint(0, 3)) if(stringy)
                           else random.randint(0, 5)
                           for x in range(sample_size)])

    df = pd.DataFrame({"true_label": one_set(sample_size, stringy),
                       "prediction_labels": one_set(sample_size, stringy),
                       "matched": [np.random.choice([0, 1], p=[0.1, 0.9]) for x in range(sample_size)]})

    df.index = [x for x in range(sample_size)]

    other_df = df[['true_label']].copy()
    other_df['prediction_labels'] = one_set(sample_size, stringy)
    other_df['matched'] = [np.random.choice([0, 1], p=[0.1, 0.9]) for x in range(sample_size)]

    return df, other_df


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
    :param on_index: bool, whether to join using series indexes or just assume ordered
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

    return results


class PipelineCompare:
    def __init__(self, experiment_path):

        base_path = path.join(experiment_path, 'qa')
        config_path = path.join(base_path, 'config.json')

        if not path.exists(base_path):
            makedirs(base_path)

        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        # Filepath to own ML results, and if desired, to comparison ML results
        self._ml_file_path = self.__config.get('ml_file_path', None)
        self._comparison_ml_file_path = self.__config.get('comparison_ml_file_path', None)

        # If the pad options are specified, then do so.
        self._pad_char = self.__config.get("pad_char", None)
        self._pad_len = self.__config.get("pad_len", None)

        self._matched_only = self.__config.get("matched_only", None)
        self._output_dir = self.__config["output_dir"]

    def expanded_right_pad(self, x):
        """
        Right-pads to config-specified length and character, hard-coded handling of the two non-coding signals
        expected in the SIC SOC project
        :param x: any single item that can be reliably cast to string
        :return: standardised string of min length _pad_len
        """
        return str(x).ljust(self._pad_len, self._pad_char)\
                          .replace("-100", "-1")\
                          .replace("-600", "-6")

    def prep_ml_table(self, df):
        """
        Some standard cleaning up for output predictions/validation files from the ml_pipeline
        :param df: predictions DataFrame with columns true_label, prediction_labels, and matched if intending to subset
        to records that passed the thresholding.
        :return: two series for true and predicted values.  Returned as series to preserve index.
        """
        if self._matched_only:
            df = df[df['matched'] == 1]

        if self._pad_char:
            y_true = df['true_label'].apply(self.expanded_right_pad)
            y_pred = df['prediction_labels'].apply(self.expanded_right_pad)

        else:
            y_true = df['true_labels']
            y_pred = df['prediction_labels']

        return y_true, y_pred

    def create_qa_outputs(self, ml_df=None):
        """
        Creates a classification table showing the results per class of the ml coding tool, and optionally a table
        comparing the results to that of another classification solution, if specified.
        :param ml_df: optionally feed an appropriately labelled DataFrame to this function, if not specifies it tries
        to load the csv specified in the config file.
        :return: None
        """

        # If no df provided, assume we're loading one saved elsewhere
        if not ml_df:
            ml_df = pd.read_csv(self._ml_file_path, index_col=0)

        y_true, y_pred = self.prep_ml_table(ml_df)

        class_df = by_class_results(y_true, y_pred)

        class_df.to_csv(path.join(self._output_dir, "by_class_results.csv"))

        if self._comparison_ml_file_path:
            other_df = pd.read_csv(self._comparison_ml_file_path)
            y_true_other, y_pred_other = self.prep_ml_table(other_df)
            compare_df = by_class_compare(y_pred, y_pred_other, y_true, on_index=True)

            compare_df.to_csv(path.join(self._output_dir, "ml_system_comparison.csv"))

        return None


# Temp scripting while I get this show on the road
df, other_df = fake_results(200, stringy=True)

print(df)
print(other_df)

df.to_csv("experiment/ml1.csv", index=True)
other_df.to_csv("experiment/ml2.csv", index=True)

configure_pipeline(experiment_path="experiment",
                   ml_file_path="experiment/ml1.csv",
                   comparison_ml_file_path="experiment/ml2.csv",
                   pad_char="0",
                   pad_len=4,
                   matched_only=False)

PC = PipelineCompare("experiment")

PC.create_qa_outputs()