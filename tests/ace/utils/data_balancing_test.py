import unittest

import numpy as np

from ace.utils.utils import DataBalancer


def fake_unbalanced_data(sample_size=20, stringy=True, random_state=42):
    """
    Generate a fake unbalanced dataset with fake features and fake labels.  Fake news!

    :param sample_size: Number of samples to fake
    :param stringy: Whether the class labels should be string or int
    :return: X, y: Arrays of fake (numeric) features and of string or int labels respectively
    """
    def _fake_up_labels(sample_size, stringy):
        """ Generate the class labels with certain probabilities. """
        options = ["C" + str(x) if(stringy) else x for x in range(8)]
        probabilities = [0.05, 0.04, 0.01, 0.4, 0.05, 0.4, 0.01, 0.04]

        return np.random.choice(a=options, size=sample_size, p=probabilities)

    np.random.seed(random_state)
    X = np.random.rand(sample_size, 10)
    y = _fake_up_labels(sample_size, stringy)

    return X, y


class DataBalancerTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.__X, self.__y = fake_unbalanced_data(6000)

    def test_transformer_runs(self):
        DB = DataBalancer()
        X_new, y_new = DB.fit_transform(self.__X, self.__y)

        return True