import pandas as pd
import pickle
import bz2

from ace.pipelines.pipeline_text import *

test = pd.read_excel("data/lcf.xlsx")

test['single_text'] = test['RECDESC'].astype(str) + " " + test['EXPDESC'].astype(str)

with bz2.BZ2File("data/proto_dat.pkl.bz2", 'wb') as pickle_file:

    pkl_obj = [list(test['single_text']), list(test['EFSCODE'])]
    pickle.dump(pkl_obj, pickle_file, protocol=4, fix_imports=False)

configure_pipeline(data_path='data/proto_dat.pkl.bz2', experiment_path=path.join('outputs', 'soc'))
pt = PipelineText(config_path=path.join('outputs', 'soc', 'text'))
test = pt.fit_transform()

print(test)

# with bz2.BZ2File("data/proto_dat.pkl.bz2", 'rb') as pickle_file:
#     X, y = pickle.load(pickle_file)
#
# print(X)
# print(LemmaTokenizer()(X[0]))
# print(StemTokenizer()(X[1]))
#
# X = SpellCheckDoc().fit_transform(X)
# print(X)
#
# X = SplitWords().fit_transform(X)
# print(X)
#
# # These are all the corpora NLTK needs to work
# # nltk_corps = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
#
# print("done!")
