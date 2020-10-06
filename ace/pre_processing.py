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
from datetime import datetime
import string
import re
from collections import Counter
from os import path
import ast

import pandas as pd
import wordninja
from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker

from ace.parser import LoadParseDicts
    
def Exceptions(code, exceptions, suffix_list = None):
    
    '''
        Loads a list of expcetion words and suffixs not to use when apply parsing function. 
        Load the lists from pickled dictionaries or from .txt files. 
    
    '''
    #Load expcetions folder 
    parser_dir = f"./ace/data_parser/{code}/"
    #if there are dictionaries available
    if exceptions is not None:
        #load list from text file
        if exceptions.endswith(".txt"):  
            with open(parser_dir + exceptions) as f: 
                exceptions = [line.strip() for line in f]
        
        #load list from dictionaries
        else:
            #load dictionaries using parser class
            dictionaries = LoadParseDicts(code).load_dicts()
            
            # get dictionary containing list of suffix excpetions
            if suffix_list is not None: 
                exceptions = tuple(dictionaries[exceptions])

            #get dictionary containing word exceptions
            else:
                exceptions = [w.replace("\\1", "").replace("\\2", "") 
                for w in dictionaries[exceptions].values()]
    
    #else load an empty list
    else: 
        exceptions = []
    return exceptions
        
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
    def __init__(self, exceptions, code = None):
        self.ps = PorterStemmer()
        self.code = code
        
        try: 
            self.suffix = exceptions['suffix']
        except ValueError: 
            print("No suffix dictionary/list provided")
            self.suffix = None
        try: 
            self.except_words = exceptions["except"]
        except ValueError: 
            print("No excpetions dictionary/list provided")
            self.except_words = None
        
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['shouldv', 'youv', 'abov', 'ani', 'becau', 'becaus', 'befor', 'doe', 'dure', 'ha', 'hi', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'wa', 'whi', 'yourselv'])
        
    def __call__(self, doc):
        if self.code in ["sic", "soc"]:
            return [self.ps.stem(t) for t in word_tokenize(doc) if t not in self.stop_words or t not in Exceptions(self.code, exceptions = self.except_words) or not t.endswith(Exceptions(self.code, suffix_list = True, exceptions = self.suffix))]
        else: 
            return [self.ps.stem(t) for t in word_tokenize(doc) if t not in self.stop_words]
    
def lowercase_strip_accents_and_ownership(doc):
    lowercase_no_accents_doc = strip_accents_ascii(str(doc).lower())
    txt = lowercase_no_accents_doc.replace('"', '').replace("\'s", "").replace("\'ve", " have").replace("\'re",
                                                                                                        " are").replace(
        "\'", "").strip("`").strip()
    return txt

def stop(tokensin, unigrams, ngrams, digits=True):
    new_tokens = []
    for token in tokensin:
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


class WordAnalyzer(object):
    tokenizer = None
    preprocess = None
    ngram_range = None
    stemmed_stop_word_set_n = set(stopwords.words('english'))
    stemmed_stop_word_set_uni = set(stopwords.words('english'))

    @staticmethod
    def init(tokenizer, preprocess, ngram_range, spell_correct=False):
        WordAnalyzer.tokenizer = tokenizer
        WordAnalyzer.preprocess = preprocess
        WordAnalyzer.ngram_range = ngram_range
        WordAnalyzer.spell = spell_correct

    # Based on VectorizeMixin in sklearn text.py
    @staticmethod
    def analyzer(doc):
        """based on VectorizerMixin._word_ngrams in sklearn/feature_extraction/text.py,
        from scikit-learn; extended to prevent generation of n-grams containing stop words"""
        min_n, max_n = WordAnalyzer.ngram_range
        original_tokens = WordAnalyzer.tokenizer(WordAnalyzer.preprocess(doc))
        
        tokens = original_tokens if min_n == 1 else []

        # handle token n-grams
        if max_n > 1:
            min_phrase = max(min_n, 2)
            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_phrase, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    candidate_ngram = original_tokens[i: i + n]
                    tokens_append(space_join(candidate_ngram))

        return stop(tokens, WordAnalyzer.stemmed_stop_word_set_uni, WordAnalyzer.stemmed_stop_word_set_n)
    
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
        
        #df = pd.DataFrame(x)
        #for c in df.columns:
        #    df[c] = df[c].str.translate(str.maketrans('', '', string.punctuation))
        #    df[c] = pd.Series([_string_correction(token) for token in df[c]], name=c)
        #    dataout = df[c]
        
        dataout = pd.Series(x).str.translate(str.maketrans('', '', string.punctuation))
        dataout = dataout.apply(_string_correction)
        return dataout

    
class SplitWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spell_ = SpellChecker(distance=1)
        self.__output_filedir = './config'
        self.stopwords_list_ = [x.upper() for x in set(stopwords.words('english'))]
    
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
        word_count_df = pd.DataFrame.from_dict(word_count,orient='index').reset_index()
        word_count_df.columns= ['words', 'n_appearances']

        #only keep actual words
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
        
        #df = pd.DataFrame(x)
        #for c in df.columns:
        #    df[c] = pd.Series([_join_up_words(token) for token in df[c]], name=c)
        #    dataout = df[c]
        
        dataout = pd.Series(x).apply(_join_up_words)
        dataout.to_csv("testing_spelling.csv")
        return dataout