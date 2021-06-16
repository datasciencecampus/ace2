import bz2
import json
import pickle
from os import path

import joblib
import nltk
import pandas as pd
import scipy
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import ace.utils.utils as ut
from nltk import word_tokenize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from ace.factories.embeddings_factory import EmbeddingsFactory
from ace.factories.feature_selection_factory import FeatureSelectionFactory
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
import numpy as np


def configure_pipeline(experiment_path,  feature_set=['frequency_matrix'], num_features=0, idf=False,
                       feature_selection_type='Logistic', min_df=3, min_ngram=1, max_ngram=3,
                       bert_name='sentence-transformers/stsb-bert-large'):
    base_path = path.join(experiment_path, 'features')
    config_path = path.join(base_path, 'config.json')
    pipe_path = path.join(base_path, 'pipe')
    data_path = path.join(experiment_path, 'text')
    d = {
        'feature_set': feature_set,
        'num_features': num_features,
        'max_df': 0.3,
        'idf': idf,
        'bert_model_name': bert_name,
        'feature_selection_type': feature_selection_type,
        'min_df': min_df,
        'min_ngram': min_ngram,
        'max_ngram': max_ngram,
        'data_path': data_path,
        'pipe_path': pipe_path,
        'base_path': base_path,
        'feature_pipeline_pickle_name':'feature_pipeline.pickle',
        'frequency_matrix': True if 'frequency_matrix' in feature_set else False,
        'sbert': True if 'sbert' in feature_set else False,
        'word_count': True if 'word_count' in feature_set else False,
        'pos': True if 'pos' in feature_set else False,
        'nmf': True if 'nmf' in feature_set else False,
        'tf_idf_filename': None
    }

    ut.check_and_create(base_path)
    with open(config_path, 'w') as fp:
        json.dump(d, fp)
    print()

class ACETransformers:
    def __init__(self, model_name):
        print('transforming text to sbert vectors...')
        # Load AutoModel from huggingface model repository
        self.__model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return None

    def transform(self, X, y=None):
        print('transforming text to sbert vectors...')
        return self.__model.encode(X)



class PipelineFeatures:
    def __init__(self, experiment_path, data_name):
        base_path = path.join(experiment_path, 'features')
        config_path = path.join(base_path, 'config.json')

        ut.check_and_create(base_path)
        with open(config_path, 'r') as fp:
            self.__config = json.load(fp)

        self.__TFIDF_FILENAME = 'tfidf.pickle'

        self.__pipeline_steps = []

        # to be changed
        # if 'embeddings' in feature_set:
        #     print("Adding embeddings!")
        #     self.__add_embeddings_features_mean(self.__text, 'fasttext_mean_300d')
        #     self.__add_embeddings_features_mean(self.__text, 'glove_mean_300d')
        #     self.__add_embeddings_features_mean(self.__text, 'fasttext_mean_300d', idf_dict=self.__idf_dict)
        #     self.__add_embeddings_features_mean(self.__text, 'glove_mean_300d', idf_dict=self.__idf_dict)

        self.__pipe = None
        self.__n_features=0
        self.__data_name=data_name

    def extend_pipe(self, steps):

        self.__pipeline_steps.extend(steps)

    def fit_transform(self, text=None, y=None):
        self.fit(text, y)
        self.transform(text, y)

    def fit(self, X=None, y=None):
        """
        Combines preprocessing steps into a pipeline object
        """

        if X is None:
            X, y = pd.read_pickle(path.join(self.__config['data_path'], self.__data_name))

        print("Assembling base feature pipeline")
        # Term Frequency!

        num_tf_features = self.__config['num_features']
        fs_model = FeatureSelectionFactory(k=num_tf_features).get_model(self.__config['feature_selection_type'])

        count_vectorizer_tuple = ("TF", self.__get_count_vectorizer() if self.__config['frequency_matrix'] else None)
        feature_selection_model_tuple = ("FS", fs_model if num_tf_features else None)
        nmf_tuple = ('NMF',
                     NMF(n_components=50, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd') if self.__config[
                         'nmf'] else None)
        idf_tuple = ('IDF', TfidfTransformer() if self.__config['idf'] and self.__config['frequency_matrix'] else None)
        wc_tuple = ('WORD_COUNT', None) # self.__add_wordcount_features(X_i) if self.__config['word_count'] else None)
        pos_tuple = ('POS', None) # self.__add_partsofspeech_features(X_i) if self.__config['pos'] else None)
        sbert_tuple = ('sBERT', ACETransformers(self.__config['bert_model_name']) if self.__config['sbert'] else None )

        pipeline_routines = [count_vectorizer_tuple, idf_tuple, feature_selection_model_tuple,sbert_tuple, nmf_tuple, wc_tuple,
                             pos_tuple]
        self.__pipeline_steps.extend([x for x in pipeline_routines if x[1] is not None])

        pipe = Pipeline(self.__pipeline_steps)

        for X_i in X:
            name = X_i.columns[0]
            X_i = np.array(X_i.values.tolist()).ravel()
            print("Fitting pipeline!")
            pipe.fit(X_i, y)

            print("Saving features pipeline")
            features_pipeline_location = path.join(self.__config['pipe_path'], self.__config['feature_pipeline_pickle_name']+'.'+ name)
            ut.check_and_create(self.__config['pipe_path'])
            joblib.dump(pipe, features_pipeline_location, compress=3)

    def transform(self, X=None, y=None):

        if X is None:
            X, y = pd.read_pickle(path.join(self.__config['data_path'], self.__data_name))
        X_list = []
        for X_i in X:
            name = X_i.columns[0]
            X_i = np.array(X_i.values.tolist()).ravel()

            features_pipeline_location = path.join(self.__config['pipe_path'],
                                                   self.__config['feature_pipeline_pickle_name']+'.'+ name)

            print("Loading feature pipeline")
            pipe = joblib.load(features_pipeline_location)

            print("Transforming data")
            X_list.append(pipe.transform(X_i))

            # # Reattach sample id's
            # if X_i.index is not None:
            #     X_i.index = X_i.index

        X_0 = X_list[0]
        for X_i in X_list[1:]:
            X_0 = scipy.sparse.hstack([X_0, X_i])
        self.__n_features = X_0.shape[1]
        self.cache_features(X_0,y, self.__data_name)
        return X_0

    def cache_features(self, X, y,suffix):

        file_name = '_xy_' + suffix

        file_name_base = self.__config['base_path']
        filename_pickle = path.join(file_name_base, file_name )

        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump((X,y), pickle_file, protocol=4, fix_imports=False)
        self.__generate_report(suffix)

    def extend_features(self, Xin, feature_columnsIn):
        for feature_columnIn in feature_columnsIn:
            Xin = self.__add_feature_to_sparcemat( Xin, feature_columnIn)
        return Xin

    def __add_feature_to_sparcemat(self, Xin, feature_columnIn):
        return scipy.sparse.hstack((csr_matrix(feature_columnIn).T, Xin))

    def __get_count_vectorizer(self):
        count_vectorizer = CountVectorizer(
            max_df=self.__config['max_df'],
            min_df=self.__config['min_df'],
            ngram_range=(self.__config['min_ngram'], self.__config['max_ngram']),
            tokenizer=word_tokenize,
            stop_words=[]
        )
        return count_vectorizer

    def __calculate_word_count(self, docs):
        return [len(str(x).split()) for x in docs]

    # def __add_wordcount_features(self, text):
    #     x_count = self.__calculate_word_count(text)
    #     self.__num_word_count = len(x_count)
    #     self.__X = self.__add_feature_to_sparcemat(self.__X, x_count)
    #     self.__X = self.__X.tocsr()

    def __add_embeddings_features_mean(self, text, model_type, idf_dict=None):
        print('calculating word vector representations: ' + model_type + ' | idf: ' + str(idf_dict is not None))
        met = EmbeddingsFactory().get_model(model_type, idf_dict=idf_dict)
        X_train_trans = met.fit_transform(text)
        self.__embeddings_counts.append(len(X_train_trans))

        for row in X_train_trans:
            self.__X = self.__add_feature_to_sparcemat(self.__X, row)
        self.__X = self.__X.tocsr()

    def __add_partsofspeech_features(self, text):
        print('calculating part of speach features...')
        pos_counts = self.__count_parts_of_speech(text)
        for x in pos_counts:
            self.__X = self.__add_feature_to_sparcemat(self.__X, x)
            self.__X = self.__X.tocsr()




    def __count_parts_of_speech(self, docs):
        # https://medium.freecodecamp.org/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24
        tag_types = ['NN', 'VB', 'JJ', 'RB']

        pos_counts = []
        for text in docs:
            tokens = nltk.word_tokenize(str(text).lower())
            text = nltk.Text(tokens)
            tagged = nltk.pos_tag(text)

            counts = nltk.Counter(tag for word, tag in tagged)
            total = sum(counts.values())
            my_dict = dict((word, float(count) / total) for word, count in counts.items())

            tag_counts = []
            for x in tag_types:
                if my_dict.get(x) in tag_types:
                    tag_counts.append(my_dict.get(x))
                else:
                    tag_counts.append(0.0)
            pos_counts.append(tag_counts)

        pos_counts_transpose = list(map(list, zip(*pos_counts)))

        self.__num_pos_count = len(pos_counts_transpose)
        return pos_counts_transpose

    def __generate_report(self, suffix):

        output_text = (
                "******* Cached Features Report ************" + "\n" +
                "num final features: " + str(self.__n_features) + "\n"
                "tf: " + str(self.__config['idf']) + "\n"
                "nmf: " + str(self.__config['nmf']) + "\n"
                "word_count: " + str(self.__config['word_count']) + "\n"
                "pos: " + str(self.__config['pos']) + "\n"
                "bert: " + str(self.__config['sbert']) + "\n")

        filename_out = path.join(self.__config['base_path'], 'report' + suffix + '.txt')
        print(output_text)

        with open(filename_out, "w") as f:
            f.write(output_text)




