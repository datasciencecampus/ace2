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
    

class LemmaTokenizer(object):
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

    def __call__(self, doc):
        text = word_tokenize(doc)
        pos_tagged_tokens = pos_tag(text)
        return [self.lemmatize_with_pos(t) for t in pos_tagged_tokens]


class StemTokenizer(object):
    def __init__(self):
        self.ps = PorterStemmer()

        self.stop_words = stopwords.words('english')
        # Read from text file ad config
        self.stop_words.extend(['shouldv', 'youv', 'abov', 'ani', 'becau', 'becaus', 'befor', 'doe', 'dure', 'ha',
                                'hi', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'wa', 'whi', 'yourselv'])
        
    def __call__(self, doc):
        return [self.ps.stem(t) for t in word_tokenize(doc) if t not in self.stop_words]


def lowercase_strip_accents_and_ownership(doc):
    lowercase_no_accents_doc = strip_accents_ascii(str(doc).lower())
    return lowercase_no_accents_doc.replace('"', '').replace("\'s", "").replace("\'ve", " have").replace("\'re",
                                                                                                        " are").replace(
        "\'", "").strip("`").strip()


def stop(tokens_in, unigrams, ngrams, digits=True):
    new_tokens = []
    for token in tokens_in:
        ngram = token.split()
        if len(ngram) == 1:
            if ngram[0] not in unigrams and not ngram[0].isdigit():
                new_tokens.append(token)
        else:
            word_in_ngrams = False
            for word in ngram:
                if word in ngrams or (digits and word.isdigit()):
                    word_in_ngrams = True
                    break
            if not word_in_ngrams:
                new_tokens.append(token)
    return new_tokens

    
class SpellCheckDoc(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spell_ = SpellChecker(distance=1) 
    
    def fit(self, x, y=None):
        return self
        
    def transform(self, x):
        print("correcting spelling")
        
        def _string_correction(original_tokens):
            corrected_text = []
            mispelled_words = self.spell_.unknown(original_tokens.split())
            for word in original_tokens.split():
                if word.lower() in mispelled_words:
                    corrected_text.append(self.spell_.correction(word).upper())
                else:
                    corrected_text.append(word.upper())
            return " ".join(corrected_text)
        
        dataout = pd.Series(x).str.translate(str.maketrans('', '', string.punctuation))
        dataout = dataout.apply(_string_correction)
        return dataout

    
class SplitWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spell_ = SpellChecker(distance=1)
        self.__output_filedir = './config'
        self.stopwords_list_ = [x.upper() for x in set(stopwords.words('english'))]
        self.__lang_filepath = None
        self.language_model_ = None

    @staticmethod
    def __keep_correctly_spelled(self, original_tokens, spell):
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
    
    def fit(self, x, y=None):
        dtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.__lang_filepath = path.join(self.__output_filedir, f'my_lang_{dtime}.txt.gz')
        df = pd.DataFrame({'words' :[self.__keep_correctly_spelled(token, self.spell_) for token in x]})
        word_count = dict(Counter(" ".join(df['words']).split(" "))) 
        word_count_df = pd.DataFrame.from_dict(word_count, orient='index').reset_index()
        word_count_df.columns= ['words', 'n_appearances']

        # Only keep actual words
        word_count_df['wordlength'] = word_count_df['words'].str.len()
        word_count_df = word_count_df[(word_count_df['wordlength'] >=3) | (word_count_df['words'].isin(self.stopwords_list_))]
        word_count_df = word_count_df.sort_values('n_appearances',ascending=False).reset_index(drop=True)
        word_count_df['words'] = word_count_df['words'].str.lower()
        word_count_df['words'].to_csv(self.__lang_filepath,index=None, header=False,compression='gzip',encoding='utf-8')
        self.language_model_ = wordninja.LanguageModel(self.__lang_filepath)

        return self
        
    def transform(self, x):
        print("Finding joined up words")
        
        def _join_up_words(original_tokens):
            corrected_text = []
            mispelled_words = self.spell_.unknown(original_tokens.split())
            for word in original_tokens.split():
                if word.lower() in mispelled_words:
                    corrected_text.append(" ".join(self.language_model_.split(word)).upper())
                else:
                    corrected_text.append(word.upper())
                
            output = " ".join(corrected_text)
            output = re.sub(r"\b(\w) (?=\w\b)", r"\1", output)
            return output
        
        dataout = pd.Series(x).apply(_join_up_words)

        return dataout


class PipelineText:
    def __init__(self, config_path):

        with open(path.join(config_path, 'config.json'), 'r') as fp:
            self.__config = json.load( fp)


        self.__pipeline_steps = []
        self.__pipe=None

        # self.__labels_hist = create_load_balance_hist(y_train)

    def extend_pipe(self, steps):

        self.__pipeline_steps.extend(steps)

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        self.transform(X, y)

    def fit(self, X=None, y=None):

        if not X:
            Χ, y = pd.read_pickle(self.__config['data_path'])
        """
        Combines preprocessing steps into a pipeline object
        """

        X=X[self.__config['text_header']]
        spell = SpellCheckDoc()
        split_words = SplitWords()


        pipeline_steps = [x for x in [("SC", spell), ("SW", split_words)]]


        self.__pipeline_steps.extend(pipeline_steps)


        self.__pipe = Pipeline(self.__pipeline_steps)
        self.__pipe.fit(X, y)

    def transform(self, X=None, y=None):

        if not X:
            X,y = pd.read_pickle(self.__config('data_path'))

        text = X[self.__config['text_header']]
        print("Transforming data")
        text = self.__pipe.transform(text, y)

        X[self.__config['text_header']]=text

        file_name = '_text_'

        file_name_base = self.__config['base_path']
        filename_pickle = path.join(file_name_base, file_name + '.pkl.bz2')

        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump(X, pickle_file, protocol=4, fix_imports=False)

        #cache X?
        return X


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


#configure_pipeline(data_path='data/USPTO-random-1000.pkl.bz2', experiment_path=path.join('outputs', 'soc'))
#pt = PipelineText(config_path=path.join('outputs', 'soc', 'text'))
#pt.fit_transform()

test_text = ["Hello and welcome to another round of Calvin-Ball!",
             "The current latest rule is that you have to provide your own sports commentary!",
             "We really hope this round ends soon..."]

print(LemmaTokenizer()(test_text[0]))
print(StemTokenizer()(test_text[1]))

X = SpellCheckDoc().fit_transform(test_text)

print(X)



# These are all the corpora NLTK needs to work
# nltk_corps = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']

print("done!")