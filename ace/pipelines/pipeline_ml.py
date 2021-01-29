import json
import joblib
import matplotlib

import numpy as np
import pandas as pd

from os import path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import ace.utils.utils as ut

matplotlib.use('Agg')


def configure_pipeline(experiment_path,  multi=True, train_test_ratio=0.75, threshold=0.5, accuracy=0.9,):

    base_path = path.join(experiment_path, 'ml')
    features_path = path.join(experiment_path, 'features')
    config_path = path.join(base_path,  'config.json')
    d={
        'multi': multi,
        'base_path': base_path,
        'features_path':features_path,
        'train_test_ratio':train_test_ratio,
        'threshold': threshold,
        'accuracy':accuracy
    }
    ut.check_and_create(base_path)
    with open(config_path, 'w') as fp: json.dump(d, fp)


class MLTrainTest():
    def __init__(self, experiment_path, data_name,classifier=None):

        base_path = path.join(experiment_path, 'ml')
        config_path = path.join(base_path, 'config.json')

        ut.check_and_create(base_path)
        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        self.__classifier = classifier
        self.__name = classifier.__class__.__name__
        ratio = self.__config['train_test_ratio']
        self.__pickle_path = path.join(self.__config['base_path'], self.__name)
        X, y = pd.read_pickle(path.join(self.__config['features_path'], '_xy_' + data_name))

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y,
                                                                                        test_size=1.0 - ratio,
                                                                                        random_state=42, shuffle=True)
        self.__accuracy=self.__config['accuracy']
        self.__threshold = self.__config['threshold']
        self.__classes = []

    def fit(self, X=None, y=None):
        print("Training a " + self.__name + " with " + str(len(self.__y_train)) + " rows")
        self.__classifier.fit(self.__X_train, self.__y_train)
        joblib.dump(self.__classifier, self.__pickle_path, compress=9)
        self.__classes = self.__classifier.classes_

        # This to save training predictions
        predictions = self.__classifier.predict(self.__X_train)
        probabilities = self.__classifier.predict_proba(self.__X_train)

        pred_df = pd.DataFrame({"true_label": self.__y_train,
                                "prediction_labels": predictions,
                                "probabilities": np.max(probabilities, axis=1)})

        pred_df.to_csv(path.join(self.__config['base_path'], 'train_predictions.csv'), index=False)

        return self

    def transform(self, X, y):
        print("transforming data using: " + self.__name)
        predictions = self.__classifier.predict(self.__X_test)
        probabilities = self.__classifier.predict_proba(self.__X_test)
        thresholds = self.__create_thresholds_list(predictions, probabilities, self.__y_test)
        joblib.dump(thresholds, self.__pickle_path+'_thresholds', compress=9)
        accuracy = accuracy_score(self.__y_test, predictions)

        # This to save test predictions
        pred_df = pd.DataFrame({"true_label": self.__y_test,
                                "prediction_labels": predictions,
                                "probabilities": np.max(probabilities, axis=1)})

        pred_df.to_csv(path.join(self.__config['base_path'], 'test_predictions.csv'), index=False)

        print("Accuracy: " + str(accuracy))

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        self.transform(X, y)

    def __create_thresholds_list(self, predictions, probabilities, y):
        accuracy = self.__accuracy
        highest_threshold = self.__threshold
        if type(y) is not list:
            y = y.tolist()
        prediction_probs = probabilities.max(axis=1)

        classes_y = set(y)
        classes_train = set(self.__classes)
        all_classes = list(classes_y.union(classes_train))

        d = {}
        thresholds = {}

        for cls in all_classes:
            d[cls]=[]

        for idx, cls in enumerate(y):
            d[cls].append((prediction_probs[idx], y[idx] == predictions[idx]))

        for cls in all_classes:

            tups = d[cls]
            sorted_tups = sorted(tups, key=lambda tup: tup[0], reverse=True)
            threshold = highest_threshold
            if len(sorted_tups)==0:
                thresholds[cls] = threshold
                continue
            lastval = sorted_tups.pop()

            while lastval[1] == 0 and len(sorted_tups) > 0:
                lastval = sorted_tups.pop()
            sorted_tups.append(lastval)

            accumulator = []
            total = 0

            for tup in sorted_tups:
                total += tup[1]
                accumulator.append(total)

            for i in range(len(accumulator)):
                accumulator[i] /= i + 1

            r_tups = list(reversed(sorted_tups))
            r_accum = list(reversed(accumulator))

            for i in range(len(r_accum)):
                if r_accum[i] >= accuracy:
                    threshold = max(threshold, r_tups[i][0])
                    break
            thresholds[cls] = threshold
        for idx, cls in enumerate(all_classes):
            print(str(cls) + ': ' + str(thresholds[cls]))
        return thresholds
