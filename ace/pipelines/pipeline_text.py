"""Sections of this code are based on scikit-learn sources; scikit-learn code is covered by the following license:
New BSD License
Copyright (c) 2007–2018 The scikit-learn developers.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
import bz2
import json
import os
import pickle
import string
import re

import joblib
import wordninja

from collections import Counter
from datetime import datetime
from os import path

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

import pandas as pd

from ace.utils.utils import create_load_balance_hist, check_and_create

import nltk
nltk.download('stopwords')


def configure_pipeline(experiment_path, data_path, clean=True, spell=True, split_words=True, text_headers=['RECDESC'],
                       stop_words=True, lemmatize=False, stemm=False):
    base_path = path.join(experiment_path, 'text')
    config_path = path.join(base_path, 'config.json')
    pipe_path = path.join(base_path, 'pipe')
    d={
        'clean': clean,
        'spell': spell,
        'split_words': split_words,
        'lemmatize': lemmatize,
        'stemm': stemm,
        'pipe_path': pipe_path,
        'data_path': data_path,
        'base_path': base_path,
        'text_pipeline_pickle_name': 'text_pipeline.pickle',
        'text_headers': text_headers,
        'stop_words':stop_words
    }
    check_and_create(base_path)
    with open(config_path, mode='w+') as fp:
        json.dump(d, fp)


class Cleaner(BaseEstimator, TransformerMixin):

    @staticmethod
    def lowercase_strip_accents_and_ownership(doc):
        lowercase_no_accents_doc = strip_accents_ascii(str(doc).lower())
        return lowercase_no_accents_doc.replace('"', '').\
                                        replace("\'s", "").\
                                        replace("\'ve", " have").\
                                        replace("\'re", " are").\
                                        replace("\'", "").\
                                        strip("`").\
                                        strip()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return [self.lowercase_strip_accents_and_ownership(doc) for doc in X]


class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        print("Lemmatizing using wordnet")
        return [' '.join([self.wnl.lemmatize(t) for t in word_tokenize(doc)]) for doc in X]


class Stemmer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ps = PorterStemmer()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        print("Stemming using porter stemmer")
        return [' '.join([self.ps.stem(t) for t in word_tokenize(doc)]) for doc in X]


class StopWords(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.__stop_words = stopwords.words('english')

        with open(os.path.join('config', 'stopwords.txt')) as f:
            extra_stop_words = f.read().splitlines()
        # Read from text file ad config
        self.__stop_words.extend(extra_stop_words)

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        print("removing stopwords")

        return [' '.join([word for word in word_tokenize(doc) if word not in self.__stop_words]) for doc in X]


class SpellCheckDoc(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spell_ = SpellChecker(distance=1) 
    
    def fit(self, X=None, y=None):
        return self

    def _string_correction(self, doc):
        tokens = word_tokenize(doc)
        mispelled_words = self.spell_.unknown(tokens)
        return " ".join([self.spell_.correction(token) if
                         (token.lower() in mispelled_words) else token
                         for token in tokens])

    def transform(self, X=None, y=None):
        print("correcting spelling")
        translations = str.maketrans('', '', string.punctuation)
        return [self._string_correction(doc.translate(translations)) for doc in X]

    
class SplitWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spell_ = SpellChecker(distance=1)
        self.__output_filedir = './config'
        self.stopwords_list_ = [x.upper() for x in set(stopwords.words('english'))]
        self.__lang_filepath = None
        self.language_model_ = None

    def fit(self, X, y=None):
        dtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.__lang_filepath = path.join(f'my_lang_{dtime}.txt.gz')
        df = pd.DataFrame({'words': X})
        word_count = dict(Counter(" ".join(df['words']).split(" ")))
        word_count_df = pd.DataFrame.from_dict(word_count, orient='index').reset_index()
        word_count_df.columns= ['words', 'n_appearances']

        # Only keep actual words
        word_count_df['wordlength'] = word_count_df['words'].str.len()
        word_count_df = word_count_df[(word_count_df['wordlength'] >=3) |
                                      (word_count_df['words'].isin(self.stopwords_list_))]

        word_count_df = word_count_df.sort_values('n_appearances', ascending=False).reset_index(drop=True)
        word_count_df['words'] = word_count_df['words'].str.lower()
        word_count_df['words'].to_csv(self.__lang_filepath,
                                      index=None,
                                      header=False,
                                      compression='gzip',
                                      encoding='utf-8')

        self.language_model_ = wordninja.LanguageModel(self.__lang_filepath)

        return self
        
    def transform(self, X=None, y=None):
        print("Finding joined up words")
        
        def _join_up_words(doc):
            corrected_text = []
            tokens = word_tokenize(doc)
            mispelled_words = self.spell_.unknown(tokens)

            for token in tokens:
                if token.lower() in mispelled_words:
                    corrected_text.append(" ".join(self.language_model_.split(token)).upper())
                else:
                    corrected_text.append(token.upper())
                
            output = " ".join(corrected_text)
            output = re.sub(r"\b(\w) (?=\w\b)", r"\1", output)
            return output
        
        return [_join_up_words(tokens) for tokens in X]


class PipelineText:
    def __init__(self, experiment_path):

        base_path = path.join(experiment_path, 'text')
        config_path = path.join(base_path, 'config.json')

        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        self.__pipeline_steps = []
        self.__pipe = None

    def extend_pipe(self, steps):
        self.__pipeline_steps.extend(steps)

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X=None, y=None):

        if X is None:
            with bz2.BZ2File(self.__config['data_path'], 'rb') as pickle_file:
                X, y = pickle.load(pickle_file)

        """
        Combines preprocessing steps into a pipeline object
        """

        pipeline_steps=[]

        if self.__config['clean']:
            pipeline_steps.append(('clean', Cleaner()))
        if self.__config['spell']:
            pipeline_steps.append(('spell', SpellCheckDoc()))
        if self.__config['split_words']:
            pipeline_steps.append(('split', SplitWords()))
        if self.__config['stop_words']:
            pipeline_steps.append(('stop', StopWords()))
        if self.__config['lemmatize']:
            pipeline_steps.append(('lemmatize', Lemmatizer()))
        if self.__config['stemm']:
            pipeline_steps.append(('stemm', Stemmer()))

        self.__pipeline_steps.extend(pipeline_steps)

        pipe = Pipeline(self.__pipeline_steps)

        for header in self.__config['text_headers']:
            print("Fitting pipeline for " + header)
            X_i = X[header]
            pipe.fit(X_i, y)

            print("Saving text pipeline")
            text_pipeline_location = path.join(self.__config['pipe_path'],
                                               self.__config['text_pipeline_pickle_name'] + '.' + X_i.name)

            check_and_create(self.__config['pipe_path'])
            joblib.dump(pipe, text_pipeline_location, compress=3)

    def transform(self, X=None, y=None):

        if X is None:
            with bz2.BZ2File(self.__config['data_path'], 'rb') as pickle_file:
                X, y = pickle.load(pickle_file)

        X_list=[]
        for header in self.__config['text_headers']:
            text_pipeline_location = path.join(self.__config['pipe_path'],
                                               self.__config['text_pipeline_pickle_name'] + '.' + header)

            print("Loading text pipeline for " + header)
            pipe = joblib.load(text_pipeline_location)
            print("Transforming pipeline for " + header)
            X_i = X[header]
            X_list.append(pipe.transform(X=X_i))

        file_name_base = self.__config['base_path']
        filename_pickle = path.join(file_name_base, 'text_features.pkl.bz2')
        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump(X, pickle_file, protocol=4, fix_imports=False)

        return X_list

    # def __load_balancing_graph(self,  clf, probabilities, suffix='labels_graph',
    #                            title='Label Counts vs Max Probabilities for: ', ax1_ylabel='max probability'):
    #     classes = self.__classes
    #     out_name = path.join(self.__outputs_dir, clf + '_load_balanced')
    #     n_classes = len(classes)
    #     fig, ax1 = plt.subplots()
    #     ax2 = ax1.twinx()  # set up the 2nd axis
    #     label_counts = [self.__labels_hist[x] for x in classes]
    #     sorted_counts_indices = sorted(range(len(label_counts)), key=lambda k: label_counts[k])
    #     sorted_probs = [probabilities[x] for x in sorted_counts_indices]
    #     sorted_classes = [classes[x] for x in sorted_counts_indices]
    #     sorted_label_counts = [label_counts[x] for x in sorted_counts_indices]
    #     ax1.plot(sorted_probs)  # plot the probability thresholds line
    #     nticks = range(n_classes)
    #
    #     # the next few lines plot the fiscal year data as bar plots and changes the color for each.
    #     ax2.bar(nticks, sorted_label_counts, width=2, alpha=0.2, color='orange')
    #     ax2.grid(b=False)  # turn off grid #2
    #     ax1.set_title(title + clf)
    #     ax1.set_ylabel(ax1_ylabel)
    #     ax2.set_ylabel('Label Counts')
    #     # Set the x-axis labels to be more meaningful than just some random dates.
    #     ax1.axes.set_xticklabels(sorted_classes, rotation='vertical', fontsize=4)
    #     ax1.set_xlabel('Labels')
    #     # Tweak spacing to prevent clipping of ylabel
    #     fig.tight_layout()
    #     plt.savefig(out_name[:-4] + suffix)
    #     plt.show()