import bz2
import json
import pickle
import string
from os import path

import joblib
import nltk
import pandas as pd
import scipy
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
v

from ace.factories.embeddings_factory import EmbeddingsFactory
from ace.factories.feature_selection_factory import FeatureSelectionFactory
from ace.pipelines.pipeline_text import lowercase_strip_accents_and_ownership, Stemmer
from scipy.sparse import csr_matrix

def configure_pipeline(experiment_path, data_path, feature_set=['frequency_matrix'], num_features=3000, idf=True,
                 feature_selection_type='Logistic', min_df=3, min_ngram=1, max_ngram=3):
    base_path = path.join(experiment_path, 'features')
    config_path = path.join(base_path,  'config.json')
    pipe_path = path.join(base_path, 'pipe')
    d={
        'feature_set':feature_set,
        'num_features': num_features,
        'max_df':0.4,
        'idf': idf,
        'feature_selection_type': feature_selection_type,
        'min_df':min_df,
        'min_ngram':min_ngram,
        'max_ngram': max_ngram,
        'data_path':data_path,
        'pipe_path':pipe_path,
        'base_path':base_path,
        'frequency_matrix': True if 'frequency_matrix' in feature_set else False,
        'embeddings': True if 'embeddings' in feature_set else False,
        'word_count': True if 'word_count' in feature_set else False,
        'pos': True if 'pos' in feature_set else False,
        'nmf': True if 'nmf' in feature_set else False,
        'tf_idf_filename':None

    }
    with open(config_path, 'w') as fp: json.dump(d, fp)


class PipelineFeatures:
    def __init__(self, config_filename):

        with open(config_filename, 'w') as fp:
            self.__config = json.load( fp)


        self.__PIPE_FILENAME = 'feature_pipeline.pickle'
        self.__TFIDF_FILENAME = 'tfidf.pickle'

        self.__pipeline_steps = []


        #to be changed
        # if 'embeddings' in feature_set:
        #     print("Adding embeddings!")
        #     self.__add_embeddings_features_mean(self.__text, 'fasttext_mean_300d')
        #     self.__add_embeddings_features_mean(self.__text, 'glove_mean_300d')
        #     self.__add_embeddings_features_mean(self.__text, 'fasttext_mean_300d', idf_dict=self.__idf_dict)
        #     self.__add_embeddings_features_mean(self.__text, 'glove_mean_300d', idf_dict=self.__idf_dict)

        self.__pipe=None

    def extend_pipe(self, steps):

        self.__pipeline_steps.extend(steps)

    def fit_transform(self, text, y):
        self.fit(text, y)
        self.transform(text, y)

    def fit(self, text=None, y=None):
        """
        Combines preprocessing steps into a pipeline object
        """

        if not text:
            text, y = pd.read_pickle(self.__config('data_path'))
        print("Assembling base feature pipeline")
        # Term Frequency!

        num_tf_features = self.__config['num_features']
        fs_model = FeatureSelectionFactory(k=num_tf_features).get_model(self.__config['feature_selection_type'])

        count_vectorizer_tuple = ("TF", self.__get_count_vectorizer())
        feature_selection_model_tuple = ("FS", fs_model if self.__config['num_features'] else None)
        nmf_tuple = ('NMF',
                     NMF(n_components=50, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd') if self.__config[
                         'nmf'] else None)
        idf_tuple = ('IDF', TfidfTransformer() if self.__config['idf'] else None)
        wc_tuple = ('WORD_COUNT', self.__add_wordcount_features(text) if self.__config['word_count'] else None)
        pos_tuple = ('POS', self.__add_partsofspeech_features(text) if self.__config['pos'] else None)

        pipeline_routines = [count_vectorizer_tuple, idf_tuple, feature_selection_model_tuple, nmf_tuple, wc_tuple,
                             pos_tuple]
        self.__pipeline_steps.extend([x for x in pipeline_routines if x[1] is not None])

        pipe = Pipeline(self.__pipeline_steps)

        print("Fitting pipeline!")
        pipe.fit(text, y)

        print("Saving features pipeline")
        self.__features_pipeline_location = path.join(self.__config['pipe_path'], self.__PIPE_FILENAME)
        joblib.dump(pipe, self.__features_pipeline_location, compress=3)

    def transform(self, text=None, y=None):

        if not text:
            text, y = pd.read_pickle(self.__config('data_path'))

        print("Loading feature pipeline")
        pipe_path = path.join(self.__features_pipeline_location)
        pipe = joblib.load(pipe_path)

        print("Transforming data")
        X = pipe.transform(text)

        # Reattach sample id's
        if text.index:
            X.index = text.index

        return X

    def cache_features(self, X, suffix):

        file_name = '_xy_' + suffix

        file_name_base = self.__config['base_path']
        filename_pickle = path.join(file_name_base, file_name + '.pkl.bz2')

        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump(X, pickle_file, protocol=4, fix_imports=False)
        self.__generate_report()

    def extend_features(self, Xin, feature_columnsIn):
        for feature_columnIn in feature_columnsIn:
            self.__add_feature_to_sparcemat(self, Xin, feature_columnIn)

    def __add_feature_to_sparcemat(self, Xin, feature_columnIn):
        return scipy.sparse.hstack((csr_matrix(feature_columnIn).T, Xin))

    def __get_count_vectorizer(self):
        tokenizer_stem = Stemmer(code=self.__outdirname, exceptions=self.__exceptions)
        stop_words = stopwords.words('english')
        # read this from file
        stop_words.extend(
            ['shouldv', 'youv', 'abov', 'ani', 'becau', 'becaus', 'befor', 'doe', 'dure', 'ha', 'hi', 'onc', 'onli',
             'ourselv', 'themselv', 'thi', 'veri', 'wa', 'whi', 'yourselv'])
        count_vectorizer = CountVectorizer(
            max_df=self.__config['max_df'],
            min_df=self.__config['min_df'],
            ngram_range=(self.__config['min_ngram'], self.__config['max_ngram']),
            preprocessor=lowercase_strip_accents_and_ownership,
            tokenizer=tokenizer_stem,
            stop_words=[''.join(c for c in s if c not in string.punctuation) for s in stop_words]
        )
        return count_vectorizer

    def __calculate_word_count(self, docs):
        return [len(str(x).split()) for x in docs]
        
    def __add_wordcount_features(self, text):
        x_count = self.__calculate_word_count(text)
        self.__num_word_count = len(x_count)
        self.__X = self.__add_feature_to_sparcemat(self.__X, x_count)
        self.__X = self.__X.tocsr()

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
                "tf: " + str(self.__n_features) + "\n"
                "nmf: " + str(self.__n_features) + "\n"
                "wordcount: " + str(self.__n_features) + "\n"
                "pos: " + str(self.__n_features) + "\n"
                "embeddings: " + str(self.__n_features) + "\n")

        filename_out = path.join(self.__config['base_path'],  'report'+suffix+'.txt')
        print(output_text)

        with open(filename_out, "w") as f:
            f.write(output_text)




