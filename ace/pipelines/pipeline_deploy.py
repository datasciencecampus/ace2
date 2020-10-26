import json
import joblib
import matplotlib
from sklearn.base import BaseEstimator, TransformerMixin

matplotlib.use('Agg')
import pandas as pd

from os import path
from sklearn.metrics import  accuracy_score

matplotlib.use('Agg')


def configure_pipeline(experiment_path,  multi=True, dirname='soc', data_filename='training_data.pkl.bz2',
                  train_test_ratio=0.75,   validation_path=''):

    base_path = path.join(experiment_path, 'deploy')
    features_path = path.join(experiment_path, 'features')
    config_path = path.join(base_path,  'config.json')
    out_path = path.join(base_path, 'out')
    d={
        'multi': multi,
        'dirname': dirname,
        'base_path': base_path,
        'features_path':features_path,
        'data_filename':data_filename,
        'train_test_ratio':train_test_ratio,
        'validation_path':validation_path,
        'out_path':out_path
    }
    with open(config_path, 'w') as fp: json.dump(d, fp)


class MLDeploy(BaseEstimator, TransformerMixin):
    # TODO does this need stopwords? No, stop will be done as part of text processing
    def __init__(self, config, classifier):
        self.__classifier = classifier
        self.__name = classifier.__class__.__name__
        self.__pickle_path = path.join(config['base_path'], self.__name)
        self.__validation_path = config['validation_path']
        self.__threshold = config['threshold']
        self.__classes = []
        self.__classification_mask=[]
        self.__thresholds = joblib.load(self.__pickle_path+'_thresholds')
        self.__config=config

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):

        print("transforming data using: " + self.__name)
        classification_model = joblib.load(self.__pickle_path)

        if not X:
            X_valid, y_valid = joblib.load(path.join(self.__validation_path, '_xy_.pkl.bz2'))
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

        with open(path.join(self.__config['out_path'], 'report.txt'), "w") as f:
            f.write(write_output)