import json
import joblib
import matplotlib
from sklearn.base import BaseEstimator, TransformerMixin

from os import path
from sklearn.metrics import accuracy_score

import ace.utils.utils as ut

matplotlib.use('Agg')


def configure_pipeline(experiment_path, classifier_name, validation_path='', threshold=0.5):

    base_path = path.join(experiment_path, 'deploy')
    config_path = path.join(base_path,  'config.json')
    ml_path = path.join(experiment_path, 'ml')
    d={
        'base_path': base_path,
        'ml_path': ml_path,
        'validation_path': validation_path,
        'classifier_name': classifier_name,
        'threshold': threshold
    }

    ut.check_and_create(base_path)
    with open(config_path, 'w') as fp: json.dump(d, fp)


class MLDeploy(BaseEstimator, TransformerMixin):
    def __init__(self, experiment_path, data_name):

        base_path = path.join(experiment_path, 'deploy')
        config_path = path.join(base_path, 'config.json')

        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        self.__name = self.__config['classifier_name']
        self.__pickle_path = path.join(self.__config['ml_path'], self.__name)
        self.__validation_path = self.__config['validation_path']
        self.__threshold = self.__config['threshold']
        self.__classes = []
        self.__classification_mask=[]
        self.__thresholds = joblib.load(self.__pickle_path+'_thresholds')
        self.__data_name = data_name

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):

        print("transforming data using: " + self.__name)
        classification_model = joblib.load(self.__pickle_path)

        if not X:
            X_valid, y_valid = joblib.load(path.join(self.__validation_path, '_xy_'+self.__data_name))
        else:
            X_valid = X
            y_valid = y

        predictions = classification_model.predict(X_valid)
        probabilities = classification_model.predict_proba(X_valid)

        self.__segment(predictions, probabilities)

        if y_valid is not None:
            self.__output(y_valid, predictions)
            return y_valid, predictions, probabilities

        return predictions, probabilities

    def __segment(self, predictions, probabilities):
        prediction_probs = probabilities.max(axis=1)
        self.__all_probs = probabilities

        for idx, cls in enumerate(predictions):
            self.__classification_mask.append(prediction_probs[idx] > self.__thresholds[cls])

    def __output(self, y, predictions):

        accuracy = accuracy_score(y, predictions)
        print("Accuracy: " + str(accuracy))

        score=0
        for idx, pred in enumerate(predictions):
            if self.__classification_mask[idx]:
                score += pred==y
        matched_accuracy = score/sum(self.__classification_mask)

        n_unclassifiable = sum([x == False for x in self.__classification_mask])

        print("****************************************")
        write_output = (
                "******* Topics Classifier ************" + "\n" +
                "num validation data: " + str(len(predictions)) + "\n" +
                "unclassified: " + str(n_unclassifiable) + "\n" +
                "match-rate: " + str(1 - (n_unclassifiable / len(predictions))) + "\n" +
                "matched accuracy: " + str(matched_accuracy) + "\n" +
                "**************************************\n")

        with open(path.join(self.__config['base_path'], 'report.txt'), "w") as f:
            f.write(write_output)