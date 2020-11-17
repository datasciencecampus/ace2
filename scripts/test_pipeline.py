import bz2
import pickle
from os import path

import ace.pipelines.pipeline_text as pt
import ace.pipelines.pipeline_feature as pf
import ace.pipelines.pipeline_ml as pm
import ace.pipelines.pipeline_deploy as dl
import ace.pipelines.pipeline_compare as pc

import ace.pipelines.pipeline_data as paddy

from ace.factories.ml_factory import MLFactory
import pandas as pd
import numpy as np


pickle_file_train = path.join('data', 'processed', 'train.pkl.bz2')
pickle_file_valid = path.join('data', 'processed', 'valid.pkl.bz2')

test = pd.read_excel("data/lcf.xlsx")

msk = np.random.rand(len(test)) < 0.85

train = test[msk]
valid = test[~msk]

paddy.configure_pipeline(experiment_path="exp_1",
                         data_path="data_out.pkl",
                         drop_nans=True,
                         load_balance_ratio=None,
                         keep_headers=['RECDESC', 'EXPDESC'],
                         label_column='EFSCODE',
                         plot_classes=False,
                         drop_classes_less_than=20,
                         drop_classes_more_than=500)

pl = paddy.PipelineData("exp_1")

train_X, train_y = pl.fit_transform(train)
valid_X, valid_y = pl.transform(valid)

print(train.shape)
print(train_X.shape)

print(valid.shape)
print(valid_X.shape)

# #
# with bz2.BZ2File(pickle_file_train, 'wb') as pickle_file:
#     pkl_obj = train[['RECDESC', 'EXPDESC']], list(train['EFSCODE'])
#     pickle.dump(pkl_obj, pickle_file, protocol=4, fix_imports=False)
#
# with bz2.BZ2File(pickle_file_valid, 'wb') as pickle_file:
#     pkl_obj = valid[['RECDESC', 'EXPDESC']], list(valid['EFSCODE'])
#     pickle.dump(pkl_obj, pickle_file, protocol=4, fix_imports=False)
#
# experiment_path = 'exp_1'
# data_path = path.join('data', 'processed')
# classifier_name = 'LogisticRegression'
#
# pt.configure_pipeline(experiment_path,data_path , spell=True, split_words=True, text_headers=['RECDESC', 'EXPDESC'],
#                       stop_words=True, lemmatize=False, stemm=False)
#
# pipe_text = pt.PipelineText('exp_1', 'train.pkl.bz2')
# pipe_text.fit_transform()
#
# pipe_text = pt.PipelineText('exp_1', 'valid.pkl.bz2')
# pipe_text.transform()
#
#
# pf.configure_pipeline(experiment_path, feature_set=['frequency_matrix'], num_features=0, idf=True,
#                        feature_selection_type='Logistic', min_df=3, min_ngram=1, max_ngram=3)
#
# pipe_features = pf.PipelineFeatures('exp_1', 'train.pkl.bz2')
# pipe_features.fit_transform()
#
# pipe_features = pf.PipelineFeatures('exp_1', 'valid.pkl.bz2')
# pipe_features.transform()
#
#
# pm.configure_pipeline(experiment_path)
#
# # ---- MODEL 1 --- #
# cls = MLFactory.factory(classifier_name)
# pipe_ml = pm.MLTrainTest(experiment_path, 'train.pkl.bz2', classifier=cls)
# pipe_ml.fit_transform()
#
# dl.configure_pipeline(experiment_path, classifier_name, validation_path=experiment_path + "/features")
# ml_dep = dl.MLDeploy(experiment_path, 'valid.pkl.bz2')
#
# y_true, y_pred, y_prob = ml_dep.transform()
# print(y_true[:10], y_pred[:10])
#
# lr_ml_df = pd.DataFrame({"true_label": y_true,
#                          "prediction_labels": y_pred})
# lr_ml_df.to_csv(path.join(experiment_path, "lr_predictions.csv"))
#
#
# # ---- MODEL 2 --- #
# second_classifier_name = "RandomForestClassifier"
# cls2 = MLFactory.factory(second_classifier_name)
# pipe_ml = pm.MLTrainTest(experiment_path, 'train.pkl.bz2', classifier=cls2)
# pipe_ml.fit_transform()
#
# dl.configure_pipeline(experiment_path, second_classifier_name, validation_path=experiment_path + "/features")
# ml_dep = dl.MLDeploy(experiment_path)
#
# y_true, y_pred, y_prob = ml_dep.transform()
# print(y_true[:10], y_pred[:10])
#
# rf_ml_df = pd.DataFrame({"true_label": y_true,
#                          "prediction_labels": y_pred})
# rf_ml_df.to_csv(path.join(experiment_path, "rf_predictions.csv"))
#
#
# # --- RUN the comparison of these results --- #
# pc.configure_pipeline(experiment_path,
#                       ml_file_path=path.join(experiment_path, "lr_predictions.csv"),
#                       comparison_ml_file_path=path.join(experiment_path, "lr_predictions.csv"))
#
# pipe_pc = pc.PipelineCompare(experiment_path, 'valid.pkl.bz2')
#
# pipe_pc.create_qa_outputs()
