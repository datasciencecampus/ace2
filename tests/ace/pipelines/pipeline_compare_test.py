import unittest
import shutil

import numpy as np
import pandas as pd

from os import path, makedirs

from ace.pipelines.pipeline_compare import configure_pipeline, PipelineCompare


def fake_my_results(sample_size=20, stringy=False):
    """
    If only it were this easy.

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


class PipelineCompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Set up the test directory
        self.__experiment_dir = path.join('tests', 'exp')
        if not path.exists(self.__experiment_dir):
            makedirs(self.__experiment_dir)

        # Create fake datasets to simulate two ML system's results
        self.__df1, self.__df2 = fake_my_results(sample_size=300, stringy=True)
        self.__df1.to_csv(path.join(self.__experiment_dir, "ml1.csv"))
        self.__df2.to_csv(path.join(self.__experiment_dir, "ml2.csv"))

    def test_config_file_non_empty(self):
        configure_pipeline(experiment_path=self.__experiment_dir,
                           ml_file_path=path.join(self.__experiment_dir, "ml1.csv"),
                           comparison_ml_file_path=path.join(self.__experiment_dir, "ml2.csv"),
                           pad_char="0",
                           pad_len=4)

        config_path = path.join(self.__experiment_dir, 'qa', 'config.json')
        filesize = path.getsize(config_path)
        self.assertNotEqual(filesize,0)

    def test_pipeline_runs(self):
        PC = PipelineCompare(self.__experiment_dir)
        PC.create_qa_outputs()

        # Check a by-class results file was created
        filepath = path.join(self.__experiment_dir, 'qa', "by_class_results.csv")
        filesize = path.getsize(filepath)
        self.assertNotEqual(filesize, 0)

        # Check a by-class comparison file between two ML systems was created
        filepath = path.join(self.__experiment_dir, 'qa', "ml_system_comparison.csv")
        filesize = path.getsize(filepath)
        self.assertNotEqual(filesize, 0)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.__experiment_dir)
