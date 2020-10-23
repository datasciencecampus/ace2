import datetime
import sys
import json
import joblib
import matplotlib
from sklearn.base import BaseEstimator, TransformerMixin

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

from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


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
    # TODO does this need stopwords? No, stop will be done as part of text processing
    def __init__(self, experiment_path, classifier=None):

        base_path = path.join(experiment_path, 'ml')
        config_path = path.join(base_path, 'config.json')

        ut.check_and_create(base_path)
        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)


        self.__classifier = classifier
        self.__name = classifier.__class__.__name__
        ratio = self.__config['train_test_ratio']
        self.__pickle_path = path.join(self.__config['base_path'], self.__name)
        X, y = pd.read_pickle(path.join(self.__config['features_path'], '_xy_.pkl.bz2'))

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
        return self

    def transform(self, X, y):
        print("transforming data using: " + self.__name)
        predictions = self.__classifier.predict(self.__X_test)
        probabilities = self.__classifier.predict_proba(self.__X_test)
        thresholds = self.__create_thresholds_list(predictions, probabilities, self.__y_test)
        joblib.dump(thresholds, self.__pickle_path+'_thresholds', compress=9)
        accuracy = accuracy_score(self.__y_test, predictions)

        print("Accuracy: " + str(accuracy))

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        self.transform(X,y)

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
        

# class PipelineML:
#     def __init__(self, config_filename, classifier):
#
#         with open(config_filename, 'w') as fp:
#             self.__config = json.load( fp)
#
#         self.__classifier = classifier
#         dtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         self.__data_dir = path.join('data',dirname)
#         self.__data_filename = data_filename
#         self.__models_dir = path.join('models', dirname)
#         self.__outputs_dir = path.join('outputs', dirname, dtime)
#
#         makedirs(self.__outputs_dir, exist_ok=True)
#         self.__dirname = dirname
#         self.CL_FILENAME_PART = '.model.pickle'
#         self.__use_cached_model = use_cached_model
#
#         self.__train_test_ratio = self.__config['train_test_ratio']
#         self.__alg_types = alg_types
#         self.__multilabeled = multi
#         self.__probs = {}
#         self.__classes = {}
#         self.__labels_hist = {}
#         self.__y_test = None
#         self.__samples_df = None
#         self.__labels_df = None
#         self.__threshold=threshold
#         self.__classifiable_mask = np.array([])
#         self.__all_probs={}
#         self.__thresholds=None
#         self.__memory = memory
#         self.__accuracy = accuracy
#
#         self.__X, self.__y = pd.read_pickle(path.join('cached_features', dirname, '_xy.pkl.bz2'))
#
#
#
#     def fit(self, X, y):
#         if X is None:
#             with bz2.BZ2File(self.__config['data_path'], 'rb') as pickle_file:
#                 X, y = pickle.load(pickle_file)
#         X_train, X_test, y_train, y_test = train_test_split(self.__X, self.__y, test_size=1.0 - self.__train_test_ratio,
#                                                             random_state=42, shuffle=True)
#
#         self.__y_test = y_test
#
#         pickle_path = path.join(self.__models_dir, classifier + self.CL_FILENAME_PART)
#
#         print("Training a " + classifier + " with " + str(len(y_train)) + " rows")
#         classification_model = MLFactory.factory(classifier).fit(X_train, y_train)
#         joblib.dump(classification_model, pickle_path, compress=9)
#
#         self.__classes = classification_model.classes_
#
#
#     def transform(self, X, y):
#         id = X.id
#         classification_model = joblib.load(pickle_path)
#         print("Testing the " + classifier + " with " + str(len(y_test)) + " rows")
#         probabilities, predictions = self.__classify(X_test, classification_model, y_test, classifier)
#
#         accuracy = accuracy_score(y_test, predictions)
#         print("Accuracy: " + str(accuracy))
#         if self.__accuracy:
#             self.__thresholds = self.__create_thresholds_list(predictions, probabilities, y_test)
#         else:
#             self.__create_thresholds_list(predictions, probabilities, y_test)
#         self.__samples_df, self.__labels_df = self.__create_metrics(predictions, probabilities, y_test)
#         self.__output_graphs(classifier, y_test)
#
#         print("******* Topics Classifier Done!*******")
#         n_unclassified = sum([x == False for x in self.__classifiable])
#         write_output = {
#             'training': (
#                     "******* Topics Classifier ************" + "\n" +
#                     "num data all: " + str(self.__X.shape[0]) + "\n" +
#                     "num data train: " + str(self.__X.shape[0] * self.__train_test_ratio) + "\n" +
#                     "num data test: " + str(self.__X.shape[0] * (1 - self.__train_test_ratio)) + "\n" +
#                     "unclassified: " + str(n_unclassified) + "\n" +
#                     "match-rate: " + str(
#                 1 - (n_unclassified / ((self.__X.shape[0] * (1 - self.__train_test_ratio))))) + "\n" +
#                     "matched accuracy: " + str(
#                 accuracy_score(self.__samples_df['true_label'], self.__samples_df['prediction_labels'])) + "\n"
#                                                                                                            "overall accuracy: " + str(
#                 accuracy) + "\n" +
#                     "**************************************\n"
#             )
#         }
#
#         with open(path.join(self.__outputs_dir, 'report.txt'), "w") as f:
#             f.write(write_output['training'])
#
#
#
#
#
#
#
#
#     def validate_models(self, balanced=False):
#         dirname = 'validation'
#         file_ext = '_valid'
#         if balanced:
#             dirname+='_balanced'
#             file_ext+='_bal'
#         full_path = path.join(self.__dirname, dirname)
#         X_valid, y_valid = pd.read_pickle(path.join('cached_features', self.__dirname, f'_xy{file_ext}.pkl.bz2'))
#
#         def test_validate_models():
#             global classification_model
#             global probabilities
#             global predictions
#             global classifier
#
#             if self.__accuracy:
#                 if self.__threshold:
#                     self.__update_thresholds(y_valid)
#             for classifier in self.__alg_types:
#                 pickle_path = path.join(self.__models_dir, classifier + self.CL_FILENAME_PART)
#
#                 classification_model = joblib.load(pickle_path)
#                 print("Validating the " + classifier + " with " + str(len(y_valid)) + " rows")
#                 probabilities, predictions = self.__classify(X_valid, classification_model, y_valid, classifier)
#                 accuracy = accuracy_score(y_valid, predictions)
#
#                 print("Accuracy: " + str(accuracy))
#
#
#                 samples_df, labels_df = self.__create_metrics(predictions, probabilities, y_valid, remove_unclassifiable=True)
#             # TODO: mark those records which are unmatched
#             n_unclassifiable = sum([x == False for x in self.__classifiable])
#             print("****************************************")
#             write_output = (
#                 "******* Topics Classifier ************" + "\n" +
#                 "num validation data: " + str(X_valid.shape[0]) + "\n" +
#                 "unclassified: " + str(n_unclassifiable) + "\n" +
#                 "match-rate: " + str(1 - (n_unclassifiable/X_valid.shape[0])) + "\n" +
#                 "matched accuracy: " + str(accuracy_score(samples_df['true_label'], samples_df['prediction_labels'])) + "\n"
#                 "overall accuracy: " + str(accuracy) + "\n" +
#                 "**************************************\n")
#             with open(path.join(self.__outputs_dir, classifier + f'report{file_ext}.txt'), "w") as f:
#                 f.write(write_output)
#
#             samples_df, labels_df = self.__create_metrics(predictions, probabilities, y_valid, remove_unclassifiable = False)
#
#             samples_df.to_csv(path.join(self.__outputs_dir, classifier + file_ext +'.csv'))
#             qa_results = qa_results_by_class(
#                               code=self.__dirname,
#                               ml_file_path=path.join(self.__outputs_dir,classifier + file_ext +'.csv'),
#                               matched_only = True,
#                               weighted=True
#                          )
#
#             qa_results.to_csv(path.join(self.__outputs_dir, 'qa_results' + file_ext + '.csv'), index=False)
#
#             join_compare_systems(code=self.__dirname,
#                                  ml_file_path=path.join(self.__outputs_dir, classifier + file_ext +'.csv'),
#                                  output_path=self.__outputs_dir,
#                                  classifier=classifier,
#                                  balanced=balanced,
#                                  id_col="id")
#
#
#     def __update_thresholds(self,y_in):
#         y=y_in.tolist()
#         for cls in y:
#             if cls not in self.__thresholds:
#                 self.__thresholds[cls]=1.1
#
#
#     def __create_metrics(self, predictions, probabilities, y, remove_unclassifiable=True):
#
#
#
#         true_positive = [x == y for x, y in zip(predictions, y)]
#         self.__probs = np.array(probabilities)
#
#         d = {}
#         for i in range(len(predictions)):
#             prediction = predictions[i]
#             match = true_positive[i]
#             prediction_probability = prediction_probs[i]
#             if prediction in d:
#                 d[prediction].append((prediction_probability, match))
#             else:
#                 d[prediction] = [(prediction_probability, match)]
#
#         d_min = {}
#         d_accuracies = {}
#         d_max = {}
#         for label in d:
#             darray = d[label]
#             sorted_array = sorted(darray, key=lambda tup: tup[0], reverse=True)
#             accuracy = sum(x for _, x in sorted_array if x is True)/len(sorted_array)
#             last_prob = 0.0
#             for tup in sorted_array:
#                 if tup[1]:
#                     last_prob = tup[0]
#                 else:
#                     break
#             while label not in d_max and len(sorted_array) > 0:
#                 tup = sorted_array.pop(0)
#                 if tup[1]:
#                     d_max[label] = tup[0]
#             d_min[label] = last_prob
#             d_accuracies[label] = accuracy
#
#         # y below automatically creates a column from its index!
#         samples_df = pd.DataFrame({'prediction_probabilities': prediction_probs,                                 'prediction_labels': predictions,
#                                    'true_positives': true_positive,
#                                    'true_label': y})
#
#         if not remove_unclassifiable and self.__threshold:
#             samples_df['matched'] = 0
#             samples_df.loc[samples_df['prediction_probabilities'] >= self.__threshold, 'matched'] = 1
#
#         labels_df = pd.DataFrame({'min_probabilities': [d_min[x] if x in d_min else 0.0 for x in self.__classes],
#                                   'accuracies': [d_accuracies[x] if x in d_accuracies else 0.0 for x in self.__classes],
#                                   'max_probabilities': [d_max[x] if x in d_max else 0.0 for x in self.__classes]})
#
#         return samples_df, labels_df
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     def __save_roc_plot(self, clf, y_in):
#         classes = self.__classes
#         out_name = path.join(self.__outputs_dir, clf + '_roc')
#         probs = np.array(self.__all_probs)
#         y = label_binarize(y_in, classes=classes)
#         n_classes = len(self.__classes)
#
#         # Compute ROC curve and ROC area for each class
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         thresholds = dict()
#         all_classes = range(n_classes)
#
#         for i in all_classes:
#             ys = y[:, i]
#             probs_i = probs[:, i]
#             fpr[i], tpr[i], thresholds[i] = roc_curve(ys, probs_i)
#             roc_auc[i] = auc(fpr[i], tpr[i])
#
#         sorted_indices = sorted(range(len(roc_auc)), key=lambda k: roc_auc[k], reverse=True)
#         num_samples = min(7, n_classes)
#         min_rocs = sorted_indices[-num_samples:]
#         max_rocs = sorted_indices[:num_samples]
#         mid_rocs = sorted_indices[(n_classes//2)-(num_samples//2): (n_classes//2)+(num_samples//2)]
#         self.__plot_roc(classes, clf + ' min', fpr, out_name + ' min', probs, min_rocs, roc_auc, tpr, y)
#         self.__plot_roc(classes, clf + ' max', fpr, out_name + ' max', probs, max_rocs, roc_auc, tpr, y)
#         self.__plot_roc(classes, clf + ' mid', fpr, out_name + ' mid', probs, mid_rocs, roc_auc, tpr, y)
#
#
#
#     def __plot_roc(self, classes, clf, fpr, out_name, probs, chosen_classes, roc_auc, tpr, y):
#         lw = 2
#         n_classes = len(classes)
#         # Compute micro-average ROC curve and ROC area
#         fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), probs.ravel())
#         roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#         # Compute macro-average ROC curve and ROC area
#         # First aggregate all false positive rates
#         all_fpr = np.unique(np.concatenate([fpr[i] for i in chosen_classes]))
#         # Then interpolate all ROC curves at this points
#         mean_tpr = np.zeros_like(all_fpr)
#         for i in chosen_classes:
#             mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#         # Finally average it and compute AUC
#         mean_tpr /= n_classes
#         fpr["macro"] = all_fpr
#         tpr["macro"] = mean_tpr
#         roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#         # Plot all ROC curves
#         plt.figure()
#         plt.plot(fpr["micro"], tpr["micro"], label='micro-avg (area = {0:0.2f})'.format(roc_auc["micro"]),
#                  color='deeppink', linestyle=':', linewidth=4)
#         plt.plot(fpr["macro"], tpr["macro"], label='macro-avg (area = {0:0.2f})'.format(roc_auc["macro"]),
#                  color='navy', linestyle=':', linewidth=4)
#         colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#         for i, color in zip(chosen_classes, colors):
#             plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='class: {0} (area = {1:0.2f})'.format(classes[i],
#                                                                                                      roc_auc[i]))
#         plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC for ' + clf)
#         plt.legend(loc="lower right")
#         plt.savefig(out_name)
#         plt.show()
#
#     def __save_confidence_plot(self, df, alg):
#         out_name = path.join(self.__outputs_dir, alg + '_confidence')
#         plt.clf()
#         probabilities_tp = df[df['true_positives'] == True]['prediction_probabilities']
#         probabilities_fp = df[df['true_positives'] == False]['prediction_probabilities']
#         probabilities_tp.hist(bins=50, alpha=0.5)
#         probabilities_fp.hist(bins=50, alpha=0.65)
#
#         plt.xlabel('Confidence')
#         plt.ylabel('Number of items')
#
#         plt.title('Classification performance of {}'.format(alg))
#
#         plt.savefig(out_name)
#         plt.show()
#
#
#
#     def __output_graphs(self, classifier, y):
#         df = self.__samples_df[['prediction_probabilities', 'prediction_labels', 'true_positives',
#                                 'true_label']]
#         df.to_csv(path.join(self.__outputs_dir, classifier + '.csv'))
#
#
#     def __confidence_score(self):
#         classes = self.__classes
#         probabilities = self.__probs
#         min_probabilities = self.__labels_df['min_probabilities']
#         actual = self.__y_test
#
#         min_probabilities[min_probabilities == 0] = 1
#         min_thresh = [np.add(p, (p >= min_probabilities)) for p in probabilities]
#
#         labelled = [list(zip(classes, p)) for p in min_thresh]
#         new_predictions = [max(l, key=lambda item: item[1])[0] for l in labelled]
#         new_acc = sum([p == a for p, a in zip(new_predictions, actual)]) / len(actual) * 100
#
#         return new_acc
#
#
#     def __load_balancing_graph(self,  clf, probabilities, suffix='labels_graph',
#                                title='Label Counts vs Max Probabilities for: ', ax1_ylabel='max probability'):
#         classes = self.__classes
#         out_name = path.join(self.__outputs_dir, clf + '_load_balanced')
#         n_classes = len(classes)
#         fig, ax1 = plt.subplots()
#         ax2 = ax1.twinx()  # set up the 2nd axis
#         label_counts = [self.__labels_hist[x] for x in classes]
#         sorted_counts_indices = sorted(range(len(label_counts)), key=lambda k: label_counts[k])
#         sorted_probs = [probabilities[x] for x in sorted_counts_indices]
#         sorted_classes = [classes[x] for x in sorted_counts_indices]
#         sorted_label_counts = [label_counts[x] for x in sorted_counts_indices]
#         ax1.plot(sorted_probs)  # plot the probability thresholds line
#         nticks = range(n_classes)
#
#         # the next few lines plot the fiscal year data as bar plots and changes the color for each.
#         ax2.bar(nticks, sorted_label_counts, width=2, alpha=0.2, color='orange')
#         ax2.grid(b=False)  # turn off grid #2
#         ax1.set_title(title + clf)
#         ax1.set_ylabel(ax1_ylabel)
#         ax2.set_ylabel('Label Counts')
#         # Set the x-axis labels to be more meaningful than just some random dates.
#         ax1.axes.set_xticklabels(sorted_classes, rotation='vertical', fontsize=4)
#         ax1.set_xlabel('Labels')
#         # Tweak spacing to prevent clipping of ylabel
#         fig.tight_layout()
#         plt.savefig(out_name[:-4] + suffix)
#         plt.show()
