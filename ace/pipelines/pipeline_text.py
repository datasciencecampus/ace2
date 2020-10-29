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

from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker
from sklearn.pipeline import Pipeline

import pandas as pd

from ace.utils.utils import check_and_create, load_corrections


def configure_pipeline(experiment_path, data_path, clean=True, spell=True, split_words=True, text_headers=['RECDESC'],
                       stop_words=True, lemmatize=False, stemm=False):
    base_path = path.join(experiment_path, 'text')
    config_path = path.join(base_path, 'config.json')
    pipe_path = path.join(base_path, 'pipe')
    lang_path = path.join(base_path, 'lang')
    d={
        'clean': clean,
        'spell': spell,
        'split_words': split_words,
        'lemmatize': lemmatize,
        'stemm': stemm,
        'pipe_path': pipe_path,
        'data_path': data_path,
        'base_path': base_path,
        'lang_path': lang_path,
        'text_pipeline_pickle_name': 'text_pipeline.pickle',
        'text_headers': text_headers,
        'stop_words':stop_words
    }
    check_and_create(base_path)
    check_and_create(lang_path)
    with open(config_path, mode='w+') as fp:
        json.dump(d, fp)


class PreCleaner(BaseEstimator, TransformerMixin):
    """
    Punctuation/accent removal, lower-case, manual corrections
    I don't like relying on pandas for this but it's a hell of a lot faster.
    """
    def __init__(self, folder="lookup_files/correction_dicts"):

        self._correction_dict = {}

        # Load all corrections, compile into a single dict of pattern (key) and replacement (value)
        for value in load_corrections(folder).values():
            self._correction_dict.update(value)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Cleaning text")

        # By far the most time consuming part of this is the bulk regex replacement of the manual corrections
        return pd.Series(X).str.upper().\
                                replace(self._correction_dict, regex=True).\
                                apply(strip_accents_ascii).\
                                replace(r"[^A-Z0-9 ]", " ", regex=True).\
                                replace(r"\s+", " ", regex=True).\
                                str.strip().\
                                str.lower().values


class LemmaTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def lemmatize_with_pos(self, tag):
        if tag[1].startswith('N'):
            return self.wnl.lemmatize(tag[0], wordnet.NOUN)
        elif tag[1].startswith('J'):
            return self.wnl.lemmatize(tag[0], wordnet.ADJ)
        elif tag[1].startswith('R'):
            return self.wnl.lemmatize(tag[0], wordnet.ADV)
        elif tag[1].startswith('V'):
            return self.wnl.lemmatize(tag[0], wordnet.VERB)
        else:
            return self.wnl.lemmatize(tag[0])

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        print("Lemmatizing and tokenizing (wordnet)")
        return [[self.lemmatize_with_pos(t) for t in pos_tag(word_tokenize(doc))] for doc in X]


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

        with open(os.path.join('lookup_files', 'stopwords.txt')) as f:
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
        
    def transform(self, X=None, y=None):
        print("correcting spelling")
        
        def _string_correction(doc):
            tokens = word_tokenize(doc)
            mispelled_words = self.spell_.unknown(tokens)
            return " ".join([self.spell_.correction(token) if
                             (token.lower() in mispelled_words) else token
                             for token in tokens])

        translations = str.maketrans('', '', string.punctuation)

        return [_string_correction(doc.translate(translations)) for doc in X]

    
class SplitWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spell_ = SpellChecker(distance=1)
        self.__output_filedir = './config'
        self.stopwords_list_ = [x.upper() for x in set(stopwords.words('english'))]
        self.__lang_filepath = None
        self.language_model_ = None

    @staticmethod
    def __keep_correctly_spelled(original_tokens, spell):
        """ Only keep words that are correctly spelled
        params: 
        * original tokens: list of words
        * spell: spellchecker.SpellChecker object
        """
        corrected_text = []
        mispelled_words = spell.unknown(original_tokens.split())
        for word in original_tokens.split():
            if word.lower() not in mispelled_words:
                corrected_text.append(word.upper())
        return " ".join(corrected_text)
    
    def fit(self, X, y=None):
        dtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame({'words' :[self.__keep_correctly_spelled(token, self.spell_) for token in X]})
        word_count = dict(Counter(" ".join(df['words']).split(" "))) 
        word_count_df = pd.DataFrame.from_dict(word_count, orient='index').reset_index()
        word_count_df.columns= ['words', 'n_appearances']

        # Only keep actual words
        word_count_df['wordlength'] = word_count_df['words'].str.len()
        word_count_df = word_count_df[(word_count_df['wordlength'] >=3) |
                                      (word_count_df['words'].isin(self.stopwords_list_))]

        word_count_df = word_count_df.sort_values('n_appearances', ascending=False).reset_index(drop=True)
        word_count_df['words'] = word_count_df['words'].str.lower()

        lang_filepath = path.join(config_test['lang_path'], f'my_lang_{dtime}.txt.gz')
        word_count_df['words'].to_csv(lang_filepath,
                                      index=None,
                                      header=False,
                                      compression='gzip',
                                      encoding='utf-8')
        self.language_model_ = wordninja.LanguageModel(lang_filepath)

        return self
        
    def transform(self, X=None, y=None):
        print("Finding joined up words")
        
        def _join_up_words(document):
            corrected_text = []
            mispelled_words = self.spell_.unknown(document.split())
            # TODO: Change this to use word_tokenize() ?
            for word in document.split():
                if word.lower() in mispelled_words:
                    corrected_text.append(" ".join(self.language_model_.split(word)).upper())
                else:
                    corrected_text.append(word.upper())
                
            output = " ".join(corrected_text)
            output = re.sub(r"\b(\w) (?=\w\b)", r"\1", output)
            return output
        
        return [_join_up_words(tokens) for tokens in X]


class PipelineText:
    def __init__(self, experiment_path):

        base_path = path.join(experiment_path, 'text')
        config_path = path.join(base_path, 'config.json')
        global config_test
        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        self.__pipeline_steps = []
        self.__pipe = None
        config_test = self.__config

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
        pipeline_steps = []

        if self.__config['clean']:
            pipeline_steps.append(("clean", PreCleaner()))
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
            X_i = X[header].astype(str)
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
            X_i = X[header].astype(str)
            new_text_list = pipe.transform(X=X_i)
            df = pd.DataFrame(new_text_list, index=X[header].index,
                              columns=[header])

            X_list.append(df)

        file_name_base = self.__config['base_path']
        filename_pickle = path.join(file_name_base, 'text_features.pkl.bz2')
        pkl_obj = X_list, y
        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump(pkl_obj, pickle_file, protocol=4, fix_imports=False)

        return X_list