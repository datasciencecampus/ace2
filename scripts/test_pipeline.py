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


experiment_path = 'exp_1'
data_path = path.join(experiment_path, 'data')

test = pd.read_excel("data/lcf.xlsx")

paddy.configure_pipeline(experiment_path=experiment_path,
                         drop_nans=True,
                         load_balance_ratio=None,
                         keep_headers=['RECDESC', 'EXPDESC'],
                         label_column='EFSCODE',
                         plot_classes=False,
                         drop_classes_less_than=10,
                         drop_classes_more_than=None)

pl = paddy.PipelineData(experiment_path)

# Option 1:  Fit and Transform as with sklearn (and handle splitting and saving files yourself
test_X, test_y = pl.fit_transform(test)

print(test.shape)
print(test_X.shape)
print(len(test_y))

# Option 2:  Have the module handle the whole process including splitting off validation set
pl.split_fit_transform_save(test, outfile_name='.pkl.bz2', split_valid=0.15)

classifier_name = 'LogisticRegression'


# ---- Text processing --- #

pt.configure_pipeline(experiment_path, data_path, spell=True, split_words=True, text_headers=['RECDESC', 'EXPDESC'],
                      stop_words=True, lemmatize=False, stemm=False)

pipe_text = pt.PipelineText(experiment_path, 'train.pkl.bz2')
pipe_text.fit_transform()

pipe_text = pt.PipelineText(experiment_path, 'valid.pkl.bz2')
pipe_text.transform()


# ---- Feature engineering --- #

pf.configure_pipeline(experiment_path, feature_set=['frequency_matrix'], num_features=0, idf=True,
                       feature_selection_type='Logistic', min_df=3, min_ngram=1, max_ngram=3)

pipe_features = pf.PipelineFeatures('exp_1', 'train.pkl.bz2')
pipe_features.fit_transform()

pipe_features = pf.PipelineFeatures('exp_1', 'valid.pkl.bz2')
pipe_features.transform()


# ---- MODEL 1 --- #

pm.configure_pipeline(experiment_path)

cls = MLFactory.factory(classifier_name)
pipe_ml = pm.MLTrainTest(experiment_path, 'train.pkl.bz2', classifier=cls)
pipe_ml.fit_transform()

dl.configure_pipeline(experiment_path, classifier_name, validation_path=experiment_path + "/features")
ml_dep = dl.MLDeploy(experiment_path, 'valid.pkl.bz2')

y_true, y_pred, y_prob = ml_dep.transform()
print(y_true[:10], y_pred[:10])

pd.DataFrame({"true_label": y_true, "prediction_labels": y_pred, "probabilities": np.max(y_prob, axis=1)}).\
   to_csv(path.join(experiment_path, "lr_predictions.csv"))


# ---- MODEL 2 --- #
second_classifier_name = "RandomForestClassifier"
cls2 = MLFactory.factory(second_classifier_name)
pipe_ml = pm.MLTrainTest(experiment_path, 'train.pkl.bz2', classifier=cls2)
pipe_ml.fit_transform()

dl.configure_pipeline(experiment_path, second_classifier_name, validation_path=experiment_path + "/features")
ml_dep = dl.MLDeploy(experiment_path, 'valid.pkl.bz2')

y_true, y_pred, y_prob = ml_dep.transform()
print(y_true[:10], y_pred[:10])

pd.DataFrame({"true_label": y_true, "prediction_labels": y_pred, "probabilities": np.max(y_prob, axis=1)}).\
   to_csv(path.join(experiment_path, "rf_predictions.csv"))


# --- Comparison of these results --- #
pc.configure_pipeline(experiment_path,
                      ml_file_path=path.join(experiment_path, "lr_predictions.csv"),
                      comparison_ml_file_path=path.join(experiment_path, "rf_predictions.csv"))

pipe_pc = pc.PipelineCompare(experiment_path)

pipe_pc.create_qa_outputs()
