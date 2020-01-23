"""
Integration tests running entire pipelines and using configuration files.
"""

import numpy
import os
import pandas as pd
import random
import unittest
from cryptonalysis.config import load_cryptonalysis_config
from cryptonalysis.ml_core.prediction import run_prediction_simulation
from cryptonalysis.ml_core.preprocessing import run_preprocessing_pipeline, CRYPTOCURRENCIES
from cryptonalysis.ml_core.training import run_training_pipeline
from cryptonalysis.ml_core.training_dl import get_split_dfs, run_deep_learning_pipeline
from datetime import date

RES_DIR = os.path.join('tests', 'resources')
CONFIG_TEST = 'config_test.conf'


class IntegrationTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(IntegrationTests, self).__init__(*args, **kwargs)
        self.config = load_cryptonalysis_config(RES_DIR, CONFIG_TEST)

        # Always have the same random arrangements
        random.seed(1)
        numpy.random.seed(1)

        # Preprocess data
        self.preprocessed_df = run_preprocessing_pipeline(self.config.crypto, self.config.preprocessing,
                                                          self.config.save_preprocessing_data, False,
                                                          master_data_dir=RES_DIR, preprocessed_data_dir=RES_DIR,
                                                          scaler_dir=RES_DIR)

    def test_preprocessing_pipeline(self):
        self.assertEqual(self.preprocessed_df.shape, (49, 11))
        self.assertEqual(self.preprocessed_df.index[0], pd.Timestamp('2019-01-10 00:00:00'))
        self.assertEqual(self.preprocessed_df.index[-1], pd.Timestamp('2019-02-27 00:00:00'))
        self.assertAlmostEquals(self.preprocessed_df.iloc[0, 0], 0.68183, 5)
        self.assertAlmostEquals(self.preprocessed_df.iloc[48, 0], 0.78106, 5)
        self.assertAlmostEquals(self.preprocessed_df.iloc[0, 9], 0.44397, 5)
        self.assertAlmostEquals(self.preprocessed_df.iloc[48, 9], 0.58220, 5)

        expected_transactions = [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
                                 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
        actual_transactions = list(self.preprocessed_df['transaction'])
        self.assertListEqual(expected_transactions, actual_transactions)

    def test_training_pipeline(self):
        trained_classifiers = run_training_pipeline(self.preprocessed_df, self.config.training)
        self.assertAlmostEqual(trained_classifiers['SVC']['training_acc'], 0.625, 3)
        self.assertEquals(trained_classifiers['SVC']['eval_acc'], 0.44)
        self.assertGreater(trained_classifiers['MLPClassifier']['training_acc'], 0)
        self.assertGreater(trained_classifiers['MLPClassifier']['eval_acc'], 0)

    def test_deep_learning_pipeline(self):
        preprocessed_dfs = {
            crypto_name: run_preprocessing_pipeline(crypto_name, self.config.preprocessing, False, False,
                                                    master_data_dir=RES_DIR, preprocessed_data_dir=RES_DIR,
                                                    scaler_dir=RES_DIR)
            for crypto_name in CRYPTOCURRENCIES
        }
        split_dfs = get_split_dfs(preprocessed_dfs, self.config.training)
        model_dict = run_deep_learning_pipeline(self.config.crypto, split_dfs, self.config.deep_learning)
        self.assertAlmostEqual(model_dict['metrics']['acc'], 0.4, 1)
        self.assertGreater(model_dict['metrics']['loss'], 0)

    def test_prediction_simulation(self):
        trained_classifiers = run_training_pipeline(self.preprocessed_df, self.config.training)
        predictor = run_prediction_simulation(self.config, RES_DIR, RES_DIR, RES_DIR, models=trained_classifiers)
        self.assertAlmostEqual(predictor.cash, 182.65191, 5)
        self.assertEquals(predictor.owned_crypto, 0)
        self.assertEquals(predictor.total_investment, 200)
        self.assertEquals(predictor.ending_date, date(2019, 9, 30))
