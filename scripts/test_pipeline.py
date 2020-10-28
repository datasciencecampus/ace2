import bz2
import pickle
from os import path

import ace.pipelines.pipeline_text as pt
import ace.pipelines.pipeline_feature as pf
import ace.pipelines.pipeline_ml as pm
import ace.pipelines.pipeline_deploy as dl
import ace.pipelines.pipeline_compare as pc

from ace.factories.ml_factory import MLFactory
import pandas as pd


pickle_file = path.join('data', 'processed', 'lcf.pkl.bz2')

# test = pd.read_excel("data/lcf.xlsx")[['RECDESC', 'EXPDESC', 'EFSCODE']].dropna()
#
# with bz2.BZ2File(pickle_file, 'wb') as pickle_file:
#     pkl_obj = test[['RECDESC', 'EXPDESC']], list(test['EFSCODE'])
#     pickle.dump(pkl_obj, pickle_file, protocol=4, fix_imports=False)

experiment_path = 'exp_1'
data_path = path.join('data', 'processed', 'lcf.pkl.bz2')
classifier_name = 'LogisticRegression'

pt.configure_pipeline(experiment_path,data_path , spell=True, split_words=True, text_headers=['RECDESC', 'EXPDESC'],
                      stop_words=True, lemmatize=False, stemm=False)

pipe_text = pt.PipelineText('exp_1')
pipe_text.fit_transform()


pf.configure_pipeline(experiment_path, feature_set=['frequency_matrix'], num_features=0, idf=True,
                       feature_selection_type='Logistic', min_df=3, min_ngram=1, max_ngram=3)

pipe_features = pf.PipelineFeatures(experiment_path)
pipe_features.fit_transform()


pm.configure_pipeline(experiment_path)

# ---- MODEL 1 --- #
cls = MLFactory.factory(classifier_name)
pipe_ml = pm.MLTrainTest(experiment_path, classifier=cls)
pipe_ml.fit_transform()

dl.configure_pipeline(experiment_path, classifier_name, validation_path=experiment_path + "/features")
ml_dep = dl.MLDeploy(experiment_path)

y_true, y_pred, y_prob = ml_dep.transform()
print(y_true[:10], y_pred[:10])

lr_ml_df = pd.DataFrame({"true_label": y_true,
                         "prediction_labels": y_pred})
lr_ml_df.to_csv(path.join(experiment_path, "lr_predictions.csv"))


# ---- MODEL 2 --- #
second_classifier_name = "RandomForestClassifier"
cls2 = MLFactory.factory(second_classifier_name)
pipe_ml = pm.MLTrainTest(experiment_path, classifier=cls2)
pipe_ml.fit_transform()

dl.configure_pipeline(experiment_path, second_classifier_name, validation_path=experiment_path + "/features")
ml_dep = dl.MLDeploy(experiment_path)

y_true, y_pred, y_prob = ml_dep.transform()
print(y_true[:10], y_pred[:10])

rf_ml_df = pd.DataFrame({"true_label": y_true,
                         "prediction_labels": y_pred})
rf_ml_df.to_csv(path.join(experiment_path, "rf_predictions.csv"))


# --- RUN the comparison of these results --- #
pc.configure_pipeline(experiment_path,
                      ml_file_path=path.join(experiment_path, "lr_predictions.csv"),
                      comparison_ml_file_path=path.join(experiment_path, "lr_predictions.csv"))

pipe_pc = pc.PipelineCompare(experiment_path)

pipe_pc.create_qa_outputs()
