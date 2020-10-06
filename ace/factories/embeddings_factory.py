from ace.mean_embedding_transformer import MeanEmbeddingTransformer


class EmbeddingsFactory(object):
    @staticmethod
    def get_model(type, idf_dict=None):
        if type == "glove_mean_300d":
            model_file_name = 'models/glove/glove.6B.300d.txt'
            return MeanEmbeddingTransformer(model_file_name, idf_dict=idf_dict)
        elif type == "fasttext_mean_300d":
            model_file_name = 'models/fasttext/wiki-news-300d-1M-subword.vec'
            return MeanEmbeddingTransformer(model_file_name, idf_dict=idf_dict)
        else:
            raise ValueError("Bad embedding type: " + type)