import numpy as np
import pandas as pd


class DataBalancer:
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

    def fit(self, X=None, y):
        # Determine number from each class to be sampled
        dynamic_limit = int(len(y) / len(pd.unique(y)))
        self._min_class_support = np.max([dynamic_limit, self._min_class_support])

        # Identify the classes that are populous enough in the fitted data to be included
        class_counts = pd.Series(y).value_counts()
        self._class_list = list(class_counts[class_counts > (2 * self._min_class_support)])

        return None

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