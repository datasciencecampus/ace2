from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
import numpy as np
import string


class MeanEmbeddingTransformer(TransformerMixin):

    def __init__(self, model_file_name, idf_dict=None):
        self.__model_file_name = model_file_name
        self.__idf_dict = idf_dict
        self._vocab, self._E = self._load_words()

    def _load_words(self):
        term_vec_dict = {}
        vocab = []
        stemmer = PorterStemmer()
        with open(self.__model_file_name, 'r', encoding="utf8") as file:
            for i, line in enumerate(file):
                line_tokens = line.split(' ')
                word = line_tokens[0]
                if word.isalpha():
                    if self.__idf_dict is not None:
                        stemmed_word = stemmer.stem(word)
                        if stemmed_word in self.__idf_dict:
                            idf = self.__idf_dict[stemmed_word]
                            vectors = [float(i) * idf for i in line_tokens[1:]]
                            term_vec_dict[word] = np.array(vectors)
                            vocab.append(word)
                    else:
                        vectors = [float(i) for i in line_tokens[1:]]
                        term_vec_dict[word] = np.array(vectors)
                        vocab.append(word)
        return np.array(vocab), term_vec_dict

    def _get_word(self, v):
        for i, emb in enumerate(self._E):
            if np.array_equal(emb, v):
                return self._vocab[i]
        return None

    def __preproc(self, docin):

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(str(docin))
        translator = str.maketrans('', '', string.punctuation)

        return [w.translate(translator) for w in word_tokens if w not in stop_words]

    def _doc_embeddings_features_extractor(self, docin):
        dok_toks = self.__preproc(docin)
        e_dok_toks = [w.lower().strip() for w in dok_toks if w.lower().strip() in self._E]
        embeddings_array = np.array([self._E[w] for w in e_dok_toks])

        try:
            mean_arr = np.nanmean(embeddings_array, axis=0)
            min_arr = np.nanmin(embeddings_array, axis=0)
            max_arr = np.nanmax(embeddings_array, axis=0)
            std_arr = np.nanstd(embeddings_array, axis=0)
            med_arr = np.nanmedian(embeddings_array, axis=0)

            ret = np.hstack((mean_arr, min_arr))
            ret = np.hstack((ret, max_arr))
            ret = np.hstack((ret, med_arr))
            ret = np.hstack((ret, std_arr))

            return ret
        except ValueError:
            pass

        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.__transpose_and_null_checks(np.array([self._doc_embeddings_features_extractor(doc) for doc in X]))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def __transpose_and_null_checks(self, lst):
        new_list = []
        for i in range(len(lst[0])):
            templist = []
            for j in range(len(lst)):
                if lst[j] is list:
                    val = lst[j][i]
                    templist.append(val)
                else:
                    templist.append(0.0)
            new_list.append(templist)
        return new_list