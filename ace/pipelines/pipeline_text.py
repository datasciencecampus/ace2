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
from matplotlib import pyplot as plt

import pandas as pd

from ace.utils.utils import create_load_balance_hist


def configure_pipeline(data_path, experiment_path, spell=True, split_words=True, text_header='RECDESC'):
    base_path = path.join(experiment_path, 'text')
    config_path = path.join(base_path, 'config.json')
    d={
        'spell': spell,
        'split_words': split_words,
        'data_path': data_path,
        'base_path': base_path,
        'text_header': text_header
    }
    if not path.exists(base_path):
        os.makedirs(base_path)
    with open(config_path, mode='w+') as fp:
        json.dump(d, fp)


class LemmaTokenizer(BaseEstimator, TransformerMixin):
    # TODO does this need stopwords?
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

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Lemmatizing and tokenizing (wordnet)")
        return [[self.lemmatize_with_pos(t) for t in pos_tag(word_tokenize(doc))] for doc in X]


class StemTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ps = PorterStemmer()

        self.stop_words = stopwords.words('english')
        # Read from text file ad config
        self.stop_words.extend(['shouldv', 'youv', 'abov', 'ani', 'becau', 'becaus', 'befor', 'doe', 'dure', 'ha',
                                'hi', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'wa', 'whi', 'yourselv'])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Stemming and tokenizing (porter)")
        return [[self.ps.stem(t) for t in word_tokenize(doc) if t not in self.stop_words] for doc in X]


class SpellCheckDoc(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spell_ = SpellChecker(distance=1) 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
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
        self.__lang_filepath = path.join(f'my_lang_{dtime}.txt.gz')
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
        word_count_df['words'].to_csv(self.__lang_filepath,
                                      index=None,
                                      header=False,
                                      compression='gzip',
                                      encoding='utf-8')

        self.language_model_ = wordninja.LanguageModel(self.__lang_filepath)

        return self
        
    def transform(self, X):
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
    def __init__(self, config_path):

        with open(path.join(config_path, 'config.json'), 'r') as fp:
            self.__config = json.load(fp)

        self.__pipeline_steps = []
        self.__pipe = None

    def extend_pipe(self, steps):

        self.__pipeline_steps.extend(steps)

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X=None, y=None):

        if not X:
            with bz2.BZ2File(self.__config['data_path'], 'rb') as pickle_file:
                X, y = pickle.load(pickle_file)

        """
        Combines preprocessing steps into a pipeline object
        """

        spell = SpellCheckDoc()
        split_words = SplitWords()
        # stem_tokenizer = StemTokenizer()
        lemma_tokenizer = LemmaTokenizer()

        # pipeline_steps = [x for x in [("SC", spell), ("SW", split_words), ("ST", stem_tokenizer)]]
        pipeline_steps = [x for x in [("SC", spell), ("SW", split_words), ("LT", lemma_tokenizer)]]

        self.__pipeline_steps.extend(pipeline_steps)

        self.__pipe = Pipeline(self.__pipeline_steps)
        self.__pipe.fit(X, y)

    def transform(self, X=None, y=None):

        if not X:
            with bz2.BZ2File(self.__config['data_path'], 'rb') as pickle_file:
                X, y = pickle.load(pickle_file)

        print("Transforming data")
        X = self.__pipe.transform(X)

        file_name = '_text_'

        file_name_base = self.__config['base_path']
        filename_pickle = path.join(file_name_base, file_name + '.pkl.bz2')

        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump(X, pickle_file, protocol=4, fix_imports=False)

        return X


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

test = pd.read_excel("../../data/lcf.xlsx")

test['single_text'] = test['RECDESC'].astype(str) + " " + test['EXPDESC'].astype(str)

with bz2.BZ2File("../../data/proto_dat.pkl.bz2", 'wb') as pickle_file:

    pkl_obj = [list(test['single_text']), list(test['EFSCODE'])]
    pickle.dump(pkl_obj, pickle_file, protocol=4, fix_imports=False)

configure_pipeline(data_path='../../data/proto_dat.pkl.bz2', experiment_path=path.join('outputs', 'soc'))
pt = PipelineText(config_path=path.join('outputs', 'soc', 'text'))
test = pt.fit_transform()

print(test)