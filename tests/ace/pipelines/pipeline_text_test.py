import os
import unittest

import pandas as pd

from ace.pipelines.pipeline_text import configure_pipeline, PipelineText, Lemmatizer
import shutil


class PipelineTextTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.__experiment_dir = os.path.join('tests', 'exp')
        self.__text = pd.DataFrame.from_dict(
            columns=['text1', 'text2', 'feat1', 'label', 'expected'],
            orient='index',
            data={
                'row_1': ['Some things', 'Sample', 0.1, 111, 'Some thing'],
                'row_2': ['Something else. Or is it?', 'nothing New', 0.2, 112, 'Something else . Or is it ?'],
                'row_3': ['It rained earlier on Tuesday', 'Shame!', 0.3, 113, 'It rained earlier on Tuesday'],
                'row_4': ['It rained cats and dogs again on Tuesday', 'Shame!', 0.3, 113,
                          'It rained cat and dog again on Tuesday'],
                'row_5': ['It rained  on Tuesday', 'Shame!', 0.3, 113, 'It rained on Tuesday'],
                'row_6': ['It rained earlier on Tuesday', 'Shame!', 0.3, 113, 'It rained earlier on Tuesday'],
                'row_7': ['It dropped further on Tuesday', 'Shame!', 0.3, 113, 'It dropped further on Tuesday'],
                'row_8': ['It rained again on Tuesday', 'Shame!', 0.3, 113, 'It rained again on Tuesday'],
                'row_9': ['It poured heavily on Tuesday', 'Shame!', 0.3, 113, 'It poured heavily on Tuesday'],
                'row_10': ['It rained cats and dogs on Tuesday', 'Shame!', 0.3, 113,
                           'It rained cat and dog on Tuesday'],
                'row_11': ["It's been sunny today", 'A thing', 0.4, 114, "It 's been sunny today"],
                'row_12': ["There's been many sunny days", 'A thing', 0.4, 114, "There 's been many sunny day"],
            })

    # def test_config_file_non_empty(self):
    #     configure_pipeline(self.__experiment_dir, 'data')
    #     config_path=os.path.join(self.__experiment_dir, 'text', 'config.json')
    #     filesize = os.path.getsize(config_path)
    #     self.assertNotEqual(filesize,0)
    #
    # def test_features_dims(self):
    #     configure_pipeline(self.__experiment_dir, 'data', text_headers=['text1'])
    #     pipe_text = PipelineText(self.__experiment_dir)
    #     pipe_text.fit(self.__text, self.__text['label'])
    #     X=pipe_text.transform(self.__text, self.__text['label'])
    #     n_cols=len(X)
    #
    #     self.assertEqual(n_cols, 1)
    #
    # def test_features_dims_two_text_columns(self):
    #     configure_pipeline(self.__experiment_dir, 'data', text_headers=['text1', 'text2'])
    #     pipe_text = PipelineText(self.__experiment_dir)
    #     pipe_text.fit(self.__text, self.__text['label'])
    #     X = pipe_text.transform(self.__text, self.__text['label'])
    #     n_cols = len(X)
    #
    #     self.assertEqual(n_cols, 2)

    def test_clean(self):
        lemmy = Lemmatizer()
        text_cleaned = lemmy.transform(self.__text['text1'])

        # Convert expected values to list so data types match
        text_expected = list(self.__text['expected'])

        print(text_cleaned)
        print(text_expected)

        self.assertEqual(text_cleaned, text_expected)

    def config_location(self):
        configure_pipeline(self.__experiment_dir, 'data')
        config_path = os.path.join(self.__experiment_dir, 'text', 'config.json')

        self.assertEqual(config_path, '/tests/exp/text/config.json')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.__experiment_dir)