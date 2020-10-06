import datetime
import sys
import json
import joblib
import matplotlib
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

from scipy import interp
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import label_binarize, normalize

from ace.factories.ml_factory import MLFactory
from ace.utils.utils import create_load_balance_hist, MemoryMonitor
from scripts.qa_results import qa_results_by_class, join_compare_systems

matplotlib.use('Agg')


class HyperparameterPipeline:
    """
    Facilitiates grid-search over hyperparameters, outputs key statistics over hyperparameters
    hyperparameter: dict of {'hyperparameter': <list of values>}
    algorithm: string name of a model that can by furnished by the MLFactory class    
    """
    
    def __init__(self, algorithm, hyperparameters, dirname='soc', cv=5):
    
        dtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.__ace_dir = path.join('outputs/', dirname, dtime)
        
        makedirs(self.__ace_dir, exist_ok=True)
        
        self.__dirname = dirname
        
        self.__algorithm = algorithm
        
        # record and unpack hyperparameters
        #self.__parameter_names = list(hyperparameters.keys())
        #self.__parameter_combos = list(product(*hyperparameters.values()))
        self.__hyperparameters = hyperparameters
        self.__cv = cv
        
        self.__X, self.__y = pd.read_pickle(path.join('cached_features', dirname, '_xy.pkl.bz2'))
        
        self.__score_set = {'accuracy': make_scorer(accuracy_score),
                            'precision_macro': make_scorer(precision_score, average="macro"),
                            'recall_macro': make_scorer(recall_score, average="macro"),
                            'precision_micro': make_scorer(precision_score, average="micro"),
                            'recall_micro': make_scorer(recall_score, average="micro")}

        
    def test_parameters(self):
        """
        For now, employs sklearn's GridSearchCV.  Given speed of training and data
        size, would be smart to switch to GridSearch without CV
        """
        classification_model = MLFactory.factory(self.__algorithm)
        
        # GridSearch is set NOT to refit the best-performant combination on
        # all data, because this class is for hyperparameter exploration, not
        # model production
        
        if self.__algorithm.startswith("RandomForest"):
            clf = RandomizedSearchCV(classification_model,
                               param_distributions=self.__hyperparameters,
                               scoring=self.__score_set,
                               return_train_score=True,
                               cv=self.__cv,
                               n_jobs=2,
                               verbose=10,
                               refit=False)
        else: 
            clf = GridSearchCV(classification_model,
                     param_grid=self.__hyperparameters,
                     scoring=self.__score_set,
                     return_train_score=True,
                     cv=self.__cv,
                     n_jobs=2,
                     verbose=10,
                     refit=False)

        clf.fit(self.__X, self.__y)
        
        results = pd.DataFrame(clf.cv_results_)
        
        results.to_csv(path.join(self.__ace_dir, "parameter_search_{}.csv".format(self.__algorithm)))
        
        print(results)
        

class MLPipeline:
    def __init__(self, alg_types, multi=True, dirname='soc', 
                 output_dirname= 'soc', data_filename='training_data.pkl.bz2',
                  train_test_ratio=0.75, use_cached_model=False, base_dir='', 
                 memory = False, threshold=0.5, accuracy=0.9):

        dtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.__data_dir = path.join('data',dirname)
        self.__data_filename = data_filename
        self.__models_dir = path.join('models', dirname)
        if output_dirname is not None:
            self.__outputs_dir = path.join('outputs', output_dirname, dtime)
        else:
            self.__outputs_dir = path.join('outputs', dirname, dtime)

        makedirs(self.__outputs_dir, exist_ok=True)
        self.__dirname = dirname
        self.CL_FILENAME_PART = '.model.pickle'
        self.__use_cached_model = use_cached_model
        
        self.__train_test_ratio = train_test_ratio
        self.__alg_types = alg_types
        self.__multilabeled = multi
        self.__probs = {}
        self.__classes = {}
        self.__labels_hist = {}
        self.__y_test = None
        self.__samples_df = None
        self.__labels_df = None
        self.__threshold=threshold
        self.__classifiable_mask = np.array([])
        self.__all_probs={}
        self.__thresholds=None
        self.__memory = memory
        self.__accuracy = accuracy
               
        self.__X, self.__y = pd.read_pickle(path.join('cached_features', dirname, '_xy.pkl.bz2'))

    def __update_thresholds(self,y_in):
        y=y_in.tolist()
        for cls in y:
            if cls not in self.__thresholds:
                self.__thresholds[cls]=1.1
    
    def test_models(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.__X, self.__y, test_size=1.0 - self.__train_test_ratio,
                                                            random_state=42, shuffle=True)
        self.__labels_hist = create_load_balance_hist(y_train)
        
        self.__y_test = y_test

        def train_test_models():
            
            global classification_model
            global predictions
            global probabilites 
            
            for classifier in self.__alg_types:
                pickle_path = path.join(self.__models_dir, classifier + self.CL_FILENAME_PART)

                if self.__use_cached_model:
                    classification_model = joblib.load(pickle_path)
                else:
                    print("Training a " + classifier + " with " + str(len(y_train)) + " rows")
                    classification_model = MLFactory.factory(classifier).fit(X_train, y_train)
                    joblib.dump(classification_model, pickle_path, compress=9)

                self.__classes = classification_model.classes_

                print("Testing the " + classifier + " with " + str(len(y_test)) + " rows")
                probabilities, predictions = self.__classify(X_test, classification_model, y_test, classifier)

                accuracy = accuracy_score(y_test, predictions)
                print("Accuracy: " + str(accuracy))
                if self.__accuracy: 
                    self.__thresholds = self.__create_thresholds_list(predictions, probabilities, y_test)
                else: 
                    self.__create_thresholds_list(predictions, probabilities, y_test)
                self.__samples_df, self.__labels_df = self.__create_metrics(predictions, probabilities, y_test)
                self.__output_graphs(classifier, y_test)
            print("******* Topics Classifier Done!*******")
            n_unclassified  = sum([x == False for x in self.__classifiable])
            write_output = {
                'training' : (
                    "******* Topics Classifier ************" + "\n" +
                    "num data all: " + str(self.__X.shape[0]) + "\n" +
                    "num data train: " + str(self.__X.shape[0] * self.__train_test_ratio) + "\n" +
                    "num data test: " + str(self.__X.shape[0] * (1 - self.__train_test_ratio)) + "\n" +
                    "unclassified: " + str(n_unclassified) + "\n" +
                    "match-rate: " + str(1 - (n_unclassified/((self.__X.shape[0]* (1 - self.__train_test_ratio))))) + "\n" +
                    "matched accuracy: " + str(accuracy_score(self.__samples_df['true_label'], self.__samples_df['prediction_labels'])) + "\n" 
                    "overall accuracy: " + str(accuracy) + "\n" +
                  "**************************************\n"
                )
            }

            with open(path.join(self.__outputs_dir, 'report.txt'), "w") as f:
                f.write(write_output['training'])
        
        # Below handles reporting on memory usage, assumes only 1 model being tested?
        max_memory = "Did not run memory tests"
        if self.__memory:
            max_memory = MemoryMonitor(train_test_models, alg_type = MLFactory.factory(self.__alg_types[0]), dirname = self.__dirname, runtype = "Training", size = self.__X.shape[0] * self.__train_test_ratio).save_memory_output()
            print(f"******* Memory Report: {max_memory} *******")  
            write_output = {
                'memory' : (
                    "******* Memory Report ************" + "\n" +
                    "num data all: " + str(self.__X.shape[0]) + "\n" +
                    "num data train: " + str(self.__X.shape[0] * self.__train_test_ratio) + "\n" +
                    "num data test: " + str(self.__X.shape[0] * (1 - self.__train_test_ratio)) + "\n" +
                    "unclassified: " + str(self.__classifiable.count(False)) + "\n" +
                    "memory useage: " + str(max_memory) + "\n" +
                    "**************************************\n"
                )
            }
            with open(path.join(self.__outputs_dir, 'memory_report.txt'), "w") as f:
                f.write(write_output['memory'])
        
        else: 
            train_test_models()
        
             
    def validate_models(self, balanced=False):
        dirname = 'validation'
        file_ext = '_valid'
        if balanced:
            dirname+='_balanced'
            file_ext+='_bal'
        full_path = path.join(self.__dirname, dirname)
        X_valid, y_valid = pd.read_pickle(path.join('cached_features', self.__dirname, f'_xy{file_ext}.pkl.bz2'))

        def test_validate_models(): 
            global classification_model
            global probabilities 
            global predictions
            global classifier
        
            if self.__accuracy:
                if self.__threshold:
                    self.__update_thresholds(y_valid)
            for classifier in self.__alg_types:
                pickle_path = path.join(self.__models_dir, classifier + self.CL_FILENAME_PART)

                classification_model = joblib.load(pickle_path)
                print("Validating the " + classifier + " with " + str(len(y_valid)) + " rows")
                probabilities, predictions = self.__classify(X_valid, classification_model, y_valid, classifier)
                accuracy = accuracy_score(y_valid, predictions)

                print("Accuracy: " + str(accuracy))


                samples_df, labels_df = self.__create_metrics(predictions, probabilities, y_valid, remove_unclassifiable=True)
            # TODO: mark those records which are unmatched
            n_unclassifiable = sum([x == False for x in self.__classifiable])
            print("****************************************")
            write_output = (
                "******* Topics Classifier ************" + "\n" +
                "num validation data: " + str(X_valid.shape[0]) + "\n" +
                "unclassified: " + str(n_unclassifiable) + "\n" +
                "match-rate: " + str(1 - (n_unclassifiable/X_valid.shape[0])) + "\n" +
                "matched accuracy: " + str(accuracy_score(samples_df['true_label'], samples_df['prediction_labels'])) + "\n" 
                "overall accuracy: " + str(accuracy) + "\n" +
                "**************************************\n")
            with open(path.join(self.__outputs_dir, classifier + f'report{file_ext}.txt'), "w") as f:
                f.write(write_output)

            samples_df, labels_df = self.__create_metrics(predictions, probabilities, y_valid, remove_unclassifiable = False)
            
            samples_df.to_csv(path.join(self.__outputs_dir, classifier + file_ext +'.csv'))
            qa_results = qa_results_by_class(
                              code=self.__dirname,
                              ml_file_path=path.join(self.__outputs_dir,classifier + file_ext +'.csv'),
                              matched_only = True, 
                              weighted=True
                         )

            qa_results.to_csv(path.join(self.__outputs_dir, 'qa_results' + file_ext + '.csv'), index=False)

            join_compare_systems(code=self.__dirname,
                                 ml_file_path=path.join(self.__outputs_dir, classifier + file_ext +'.csv'),
                                 output_path=self.__outputs_dir,
                                 classifier=classifier,
                                 balanced=balanced,
                                 id_col="id")      
        
        if self.__memory:
            if balanced: 
                runtype = "Validation balanced"
            else: 
                runtype = "Validation"
                
            max_memory = MemoryMonitor(test_validate_models, alg_type = MLFactory.factory(self.__alg_types[0]), dirname = self.__dirname, runtype = runtype, size = X_valid.shape[0]).save_memory_output()        
            
            print(f"******* Memory Report: {max_memory} *******")  
            write_output = (
                    "******* Memory Report ************" + "\n" +
                    "num validation data: " + str(X_valid.shape[0]) + "\n" +
                    "unclassified: " + str(self.__classifiable.count(False)) + "\n" +
                    "match-rate: " + str(1 - (self.__classifiable.count(False)/X_valid.shape[0])) + "\n" +
                    "memory useage: " + str(max_memory) + "\n" +
                    "**************************************\n")
            with open(path.join(self.__outputs_dir, f'memory_report{file_ext}.txt'), "w") as f:
                f.write(write_output)
        else: 
            test_validate_models()

    def __create_thresholds_list(self, predictions, probabilities, y):
        accuracy=self.__accuracy
        highest_threshold=self.__threshold
        y = y.tolist()
        prediction_probs = probabilities.max(axis=1)

        classes_y = set(y)
        classes_train = set(self.__classes)
        all_classes = list(classes_y.union(classes_train))
        
        d={}
        thresholds={}
        for idx, cls in enumerate(y):
          if cls not in d:
            d[cls]=[]
          
          d[cls].append((prediction_probs[idx], y[idx]==predictions[idx]))
          
        for cls in all_classes:
          if not accuracy or cls not in d or len(d[cls]) < 500:
            thresholds[cls]=highest_threshold
            continue
          tups = d[cls]
          sorted_tups = sorted(tups, key=lambda tup: tup[0], reverse=True)

          lastval=sorted_tups.pop()
          
          while lastval[1] ==0 and len(tups)>0:
            lastval=sorted_tups.pop()
          sorted_tups.append(lastval)
          
          accumulator=[]
          total=0
          
          for tup in sorted_tups:
            total+=tup[1]
            accumulator.append(total)
          
          for i in range(len(accumulator)):
            accumulator[i]/=i+1
          
          r_tups=list(reversed(sorted_tups))
          r_accum=list(reversed(accumulator))
          
    
          threshold = highest_threshold
          for i in range(len(r_accum)):
            if r_accum[i]>=accuracy:
              threshold=r_tups[i][0]
              break
          thresholds[cls]=threshold
        for idx, cls in enumerate(all_classes):
          print (str(cls) +': '+str(thresholds[cls]))
        return thresholds

    def __create_metrics(self, predictions, probabilities, y, remove_unclassifiable=True):
        
        prediction_probs = probabilities.max(axis=1)
        self.__all_probs = probabilities
        
        if self.__threshold:
            if self.__accuracy:
                self.__classifiable = []
        
                for idx, cls in enumerate(predictions):
                    self.__classifiable.append(prediction_probs[idx] > self.__thresholds[cls])
     
            else: 
                self.__classifiable = prediction_probs > self.__threshold
        if remove_unclassifiable:
            predictions = predictions[self.__classifiable]
            probabilities = probabilities[self.__classifiable]
            y = y[self.__classifiable]
            prediction_probs = prediction_probs[self.__classifiable]
        
        true_positive = [x == y for x, y in zip(predictions, y)]
        self.__probs = np.array(probabilities)
          
        d = {}
        for i in range(len(predictions)):
            prediction = predictions[i]
            match = true_positive[i]
            prediction_probability = prediction_probs[i]
            if prediction in d:
                d[prediction].append((prediction_probability, match))
            else:
                d[prediction] = [(prediction_probability, match)]

        d_min = {}
        d_accuracies = {}
        d_max = {}
        for label in d:
            darray = d[label]
            sorted_array = sorted(darray, key=lambda tup: tup[0], reverse=True)
            accuracy = sum(x for _, x in sorted_array if x is True)/len(sorted_array)
            last_prob = 0.0
            for tup in sorted_array:
                if tup[1]:
                    last_prob = tup[0]
                else:
                    break
            while label not in d_max and len(sorted_array) > 0:
                tup = sorted_array.pop(0)
                if tup[1]:
                    d_max[label] = tup[0]
            d_min[label] = last_prob
            d_accuracies[label] = accuracy
        
        # y below automatically creates a column from its index!
        samples_df = pd.DataFrame({'prediction_probabilities': prediction_probs,                                 'prediction_labels': predictions,
                                   'true_positives': true_positive,
                                   'true_label': y})
        
        if not remove_unclassifiable and self.__threshold:
            samples_df['matched'] = 0
            samples_df.loc[samples_df['prediction_probabilities'] >= self.__threshold, 'matched'] = 1

        labels_df = pd.DataFrame({'min_probabilities': [d_min[x] if x in d_min else 0.0 for x in self.__classes],
                                  'accuracies': [d_accuracies[x] if x in d_accuracies else 0.0 for x in self.__classes],
                                  'max_probabilities': [d_max[x] if x in d_max else 0.0 for x in self.__classes]})
        
        return samples_df, labels_df

    def __classify(self, X_test, classification_model, y_test, clf_str):
        scores=[]
        predictions = classification_model.predict(X_test)
        probabilities = classification_model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, predictions)

        sensitivity = 0
        specificity = 0
        # todo for each label...then rid of multi-labeled
        #  move to metrics. Do one for each label, like accuracies..
        if not self.__multilabeled:
            cm = confusion_matrix(y_test, predictions)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            print("Sensitivity: " + str(sensitivity))
            print("Specificity: " + str(specificity))
            print("tn: " + str(tn))
            print("fp: " + str(fp))
            print("fn: " + str(fn))
            print("tp: " + str(tp))

        scores.append((accuracy, sensitivity, specificity, clf_str))
        # TODO: we do not return this scores variable anywhere
        return probabilities, predictions

    def get_prediction(self, datain, model_name):
        X_tfidf_test = self.__transform_using_saved_tfidf(datain)
        clf = joblib.load(path.join(self.__models_dir, model_name + self.CL_FILENAME_PART))
        predictions = clf.predict(X_tfidf_test)

    def __create_thresholds_array(self, fpr, tpr, thresholds):
        max_tpr = 0.0
        idx = 0
        for i in range(len(fpr)):
            fpri = fpr[i]
            if fpri == 0:
                break
            if tpr[i] > max_tpr and fpri == 0:
                max_tpr = tpr[i]
                idx = i
        return thresholds[idx]

    def __save_roc_plot(self, clf, y_in):
        classes = self.__classes
        out_name = path.join(self.__outputs_dir, clf + '_roc')
        probs = np.array(self.__all_probs)
        y = label_binarize(y_in, classes=classes)
        n_classes = len(self.__classes)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()
        all_classes = range(n_classes)

        for i in all_classes:
            ys = y[:, i]
            probs_i = probs[:, i]
            fpr[i], tpr[i], thresholds[i] = roc_curve(ys, probs_i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        sorted_indices = sorted(range(len(roc_auc)), key=lambda k: roc_auc[k], reverse=True)
        num_samples = min(7, n_classes)
        min_rocs = sorted_indices[-num_samples:]
        max_rocs = sorted_indices[:num_samples]
        mid_rocs = sorted_indices[(n_classes//2)-(num_samples//2): (n_classes//2)+(num_samples//2)]
        self.__plot_roc(classes, clf + ' min', fpr, out_name + ' min', probs, min_rocs, roc_auc, tpr, y)
        self.__plot_roc(classes, clf + ' max', fpr, out_name + ' max', probs, max_rocs, roc_auc, tpr, y)
        self.__plot_roc(classes, clf + ' mid', fpr, out_name + ' mid', probs, mid_rocs, roc_auc, tpr, y)

    def __load_balancing_graph(self,  clf, probabilities, suffix='labels_graph',
                               title='Label Counts vs Max Probabilities for: ', ax1_ylabel='max probability'):
        classes = self.__classes
        out_name = path.join(self.__outputs_dir, clf + '_load_balanced')
        n_classes = len(classes)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # set up the 2nd axis
        label_counts = [self.__labels_hist[x] for x in classes]
        sorted_counts_indices = sorted(range(len(label_counts)), key=lambda k: label_counts[k])
        sorted_probs = [probabilities[x] for x in sorted_counts_indices]
        sorted_classes = [classes[x] for x in sorted_counts_indices]
        sorted_label_counts = [label_counts[x] for x in sorted_counts_indices]
        ax1.plot(sorted_probs)  # plot the probability thresholds line
        nticks = range(n_classes)

        # the next few lines plot the fiscal year data as bar plots and changes the color for each.
        ax2.bar(nticks, sorted_label_counts, width=2, alpha=0.2, color='orange')
        ax2.grid(b=False)  # turn off grid #2
        ax1.set_title(title + clf)
        ax1.set_ylabel(ax1_ylabel)
        ax2.set_ylabel('Label Counts')
        # Set the x-axis labels to be more meaningful than just some random dates.
        ax1.axes.set_xticklabels(sorted_classes, rotation='vertical', fontsize=4)
        ax1.set_xlabel('Labels')
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.savefig(out_name[:-4] + suffix)
        plt.show()

    def __plot_roc(self, classes, clf, fpr, out_name, probs, chosen_classes, roc_auc, tpr, y):
        lw = 2
        n_classes = len(classes)
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in chosen_classes]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in chosen_classes:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], label='micro-avg (area = {0:0.2f})'.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"], label='macro-avg (area = {0:0.2f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(chosen_classes, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='class: {0} (area = {1:0.2f})'.format(classes[i],
                                                                                                     roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for ' + clf)
        plt.legend(loc="lower right")
        plt.savefig(out_name)
        plt.show()

    def __save_confidence_plot(self, df, alg):
        out_name = path.join(self.__outputs_dir, alg + '_confidence')
        plt.clf()
        probabilities_tp = df[df['true_positives'] == True]['prediction_probabilities']
        probabilities_fp = df[df['true_positives'] == False]['prediction_probabilities']
        probabilities_tp.hist(bins=50, alpha=0.5)
        probabilities_fp.hist(bins=50, alpha=0.65)

        plt.xlabel('Confidence')
        plt.ylabel('Number of items')

        plt.title('Classification performance of {}'.format(alg))

        plt.savefig(out_name)
        plt.show()
    
    def __get_key(self, item):
        return item[0]

    def __output_graphs(self, classifier, y):
        df = self.__samples_df[['prediction_probabilities', 'prediction_labels', 'true_positives',
                                'true_label']]
        df.to_csv(path.join(self.__outputs_dir, classifier + '.csv'))
        self.__save_confidence_plot(df, classifier)
        self.__save_roc_plot(classifier, y)

        self.__load_balancing_graph(classifier, self.__labels_df['max_probabilities'], suffix='max_probs_graph')
        self.__load_balancing_graph(classifier, self.__labels_df['min_probabilities'], suffix='min_probs_graph',
                                    title='Label Counts vs Min Probabilities for: ', ax1_ylabel='min probability')
        self.__load_balancing_graph(classifier, self.__labels_df['accuracies'], suffix='accuracies_graph',
                                    title='Label Counts vs Accuracies for: ', ax1_ylabel='accuracy')

    def __confidence_score(self):
        classes = self.__classes
        probabilities = self.__probs
        min_probabilities = self.__labels_df['min_probabilities']
        actual = self.__y_test

        min_probabilities[min_probabilities == 0] = 1
        min_thresh = [np.add(p, (p >= min_probabilities)) for p in probabilities]

        labelled = [list(zip(classes, p)) for p in min_thresh]
        new_predictions = [max(l, key=lambda item: item[1])[0] for l in labelled]
        new_acc = sum([p == a for p, a in zip(new_predictions, actual)]) / len(actual) * 100

        return new_acc
