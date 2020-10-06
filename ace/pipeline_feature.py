import bz2
import pickle
import string
from os import path

import joblib
import pandas as pd
from nltk import Counter, pos_tag, Text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline

from ace.factories.embeddings_factory import EmbeddingsFactory
from ace.factories.feature_selection_factory import FeatureSelectionFactory
from ace.pre_processing import WordAnalyzer, lowercase_strip_accents_and_ownership, Exceptions, StemTokenizer, SpellCheckDoc, SplitWords
from ace.parser import ParseSic, ParseSoc


class DumbFeaturesPipeline:
    def __init__(self, data_dirname, data_filename, exceptions, feature_set=['frequency_matrix'], run_type='training', num_features=3000, idf=True,
                 feature_selection_type='Logistic', spell=True, split_words=True, parser=False, parse_dir="./ace/data_parser/", min_df=3):

        # todo: write new infer / validation data app

        self.__PIPE_FILENAME = 'feature_pipeline.pickle'
        self.__TFIDF_FILENAME = 'tfidf.pickle'

        self.__ngram_range = (1, 3)
        self.__feature_set = feature_set
        self.__feature_selection_type = feature_selection_type
        self.__idf = idf
        self.__idf_dict = None
        self.__embeddings_counts = []
        self.__nmf_text = None
        self.__outdirname = path.basename(data_dirname)
        self.__num_word_features = 1
        self.__spell = spell
        self.__split_words = split_words
        self.__min_df = min_df
        self.__parser= parser
        self.__exceptions = exceptions
        
        # Feature Selection!
        if num_features != "all": 
            self.__num_features = int(num_features)
        else: 
            self.__num_features = num_features
            
        print('Reading data...')
        # collect training/test and valid set
        self.__text, self.__y = pd.read_pickle(path.join(data_dirname, data_filename))
        
        if 'embeddings' in feature_set:
            print("Adding embeddings!")
            self.__add_embeddings_features_mean(self.__text, 'fasttext_mean_300d')
            self.__add_embeddings_features_mean(self.__text, 'glove_mean_300d')
            self.__add_embeddings_features_mean(self.__text, 'fasttext_mean_300d', idf_dict=self.__idf_dict)
            self.__add_embeddings_features_mean(self.__text, 'glove_mean_300d', idf_dict=self.__idf_dict)

        # additional features
        if 'word_features' in feature_set:
            print("Adding word features!")
            self.__add_wordcount_features(self.__text)
            self.__add_partsofspeech_features(self.__text)
        else:
            self.__num_pos_count = 0
            self.__num_word_features = 0
            
        if 'nmf' in feature_set:
            self.__add_nmf_features(X)  # before feature selection
        else:
            self.__num_nmf = 0

        self.__features = self.__create_pipeline()
        self.__n_features = self.__features.shape[1]
        
    
    def __create_pipeline(self):
        """
        Combines preprocessing steps into a pipeline object
        """
        spell = None
        if self.__spell:
            spell = SpellCheckDoc()
        
        split_words = None
        if self.__split_words:
            split_words = SplitWords()
        
        parser = None
        if self.__parser:
            if self.__outdirname == "sic":
                parser = ParseSic()
            elif self.__outdirname == "soc":
                parser = ParseSoc()
            else: 
                raise ValueError("Code is not recognised. Please use either 'sic' or 'soc'")
            
        tokenizer_stem = StemTokenizer(code = self.__outdirname, exceptions = self.__exceptions)
        stop_words = stopwords.words('english')
        stop_words.extend(['shouldv', 'youv', 'abov', 'ani', 'becau', 'becaus', 'befor', 'doe', 'dure', 'ha', 'hi', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'wa', 'whi', 'yourselv'])
        
        
        print("Assembling base feature pipeline")
        # Term Frequency!
        count_vectorizer = CountVectorizer(
            max_df=0.4,
            min_df=self.__min_df,
            ngram_range=self.__ngram_range,
            preprocessor=lowercase_strip_accents_and_ownership,
            tokenizer=tokenizer_stem,
            stop_words= [''.join(c for c in s if c not in string.punctuation) for s in stop_words]
        )
        
        # Inverse Document Frequency!
        tfidf_transformer = None
        nmf_transformer = None
        if self.__idf:
            tfidf_transformer = TfidfTransformer()
            if 'nmf' in self.__feature_set:
                nmf_transformer = NMF(n_components=50, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd')
        
        feature_selection_model = FeatureSelectionFactory(k=self.__num_features).get_model(
            self.__feature_selection_type)
                   

        pipeline_steps = [x for x in [("Parser", parser),
                                      ("SC", spell),
                                      ("SW", split_words),
                                      ("TF", count_vectorizer),
                                      ("IDF", tfidf_transformer),
                                      ("NMF", nmf_transformer),
                                      ("FS", feature_selection_model)] if x[1] != None]
        
        pipe = Pipeline(pipeline_steps)
        print("Fitting pipeline!")
        transformed_X = pipe.fit_transform(self.__text, self.__y)
        
        # Reattach sample id's
        transformed_X.index = self.__text.index
        
        print("Saving features pipeline")
        file_name_base = path.join('models', self.__outdirname)
        self.__features_pipeline_location = path.join(file_name_base, self.__PIPE_FILENAME)
        joblib.dump(pipe, self.__features_pipeline_location, compress=3)
        
        return transformed_X
        
    
    def reuse_pipeline(self, new_text):
        """
        TODO: use this for validation data
        """
        print("Loading feature pipeline")
        pipe_path = path.join(self.__features_pipeline_location)
        pipe = joblib.load(pipe_path)
        
        print("Transforming data")
        X = pipe.transform(new_text)
        
        # Reattach sample id's
        X.index = new_text.index
        
        return X
            
        
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
   
    def save_data(self, file_dir, run_type, X = None, y = None):
        if X is None:
            X = self.__features
        if y is None:
            y = self.__y
            
        num_embeddings = sum(self.__embeddings_counts)
        output_text = (
                "******* Cached Features Report ************" + "\n" +
                "features: " + str(self.__feature_set) + "\n" +
                "Feature Selection: " + self.__feature_selection_type + "\n" +
                "idf: " + ('yes' if self.__idf else 'no') + "\n" +
                "spelling corrected: " + str(self.__spell) + "\n" + 
                "n-gram range: " + str(self.__ngram_range) + "\n" +
                "num classes: " + str(len(list(set(self.__y)))) + "\n" +
                "num features selection: " + str(self.__num_features) + "\n" +
                "num final features: " + str(self.__n_features) + "\n")
        if num_embeddings >3:
            output_text+=(
                    "num embeddings features: " + str(num_embeddings) + "\n" +
                    "num fast-text embeddings with idf: " + str(self.__embeddings_counts[2]) + "\n" +
                    "num fast-text embeddings no idf: " + str(self.__embeddings_counts[0]) + "\n" +
                    "num glove embeddings with idf: " + str(self.__embeddings_counts[3]) + "\n" +
                    "num glove embeddings no idf: " + str(self.__embeddings_counts[1]) + "\n")

        # output header
        file_name='_xy'
        if run_type.startswith('validation'):
            file_name+='_valid'
            if run_type.endswith('balanced'):
                file_name+='_bal'
        file_name_base = path.join('cached_features', file_dir)
        filename_pickle = path.join(file_name_base, file_name + '.pkl.bz2')
        filename_out = path.join(file_name_base, file_name +'.txt')
        print(output_text)

        with open(filename_out, "w") as f:
            f.write(output_text)
        
        if self.__nmf_text:
            with open(nmf_out, "w") as f:
                f.write(self.__nmf_text)

        obj = X, y

        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump(obj, pickle_file, protocol=4, fix_imports=False)
        with bz2.BZ2File(filename_pickle, 'wb') as pickle_file:
            pickle.dump(obj, pickle_file, protocol=4, fix_imports=False)
