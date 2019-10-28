"""
Unit tests for the training.py module.
"""

import os
import pandas as pd
import unittest
from cryptonalysis.ml_core.preprocessing import load_preprocessed_data
from cryptonalysis.ml_core.training import split_datasets, get_optimized_classifier
from cryptonalysis.ml_core.transaction_builders import BiffPredictor
from datetime import date
from sklearn import svm

RES_DIR = 'tests/resources'
HISTORICAL_TEST_DATA = 'historical_test_data.csv'
HISTORICAL_TEST_FILE = os.path.join(RES_DIR, HISTORICAL_TEST_DATA)


# Preprocessing constants for tests
CRYPTO_NAME = 'ETH'
PRICE_COLUMN = 'Close'
PREDICTOR_CLS = BiffPredictor
LOOKAHEAD_DAYS = 1
STARTING_DATE = date(2019, 1, 1)
ENDING_DATE = date(2019, 1, 20)
WINDOW_SIZE = 5
NORMALIZE = True
NORMALIZE_BY_ROW = True
STANDARDIZE = False
PROB_BUY = 1
PROB_SELL = 1

# Training constants for tests
SHUFFLE = True
CV_FOLDS = 4


class TestTrainingFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTrainingFunctions, self).__init__(*args, **kwargs)
        self.preprocessed_data = \
            load_preprocessed_data(crypto_name=CRYPTO_NAME, predictor_class=PREDICTOR_CLS,
                                   lookahead_days=LOOKAHEAD_DAYS, starting_date=STARTING_DATE, ending_date=ENDING_DATE,
                                   window_size=WINDOW_SIZE, normalize=NORMALIZE, normalize_by_row=NORMALIZE_BY_ROW,
                                   standardize=STANDARDIZE, prob_buy=PROB_BUY, prob_sell=PROB_SELL,
                                   preprocessed_data_dir=RES_DIR)

    def test_split_datasets(self):
        # Test default shuffle split
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(self.preprocessed_data, shuffle=True)
        self.assertEqual(X_dev.shape, (7, 5))
        self.assertEqual(y_dev.shape, (7,))
        self.assertListEqual(list(X_dev.index), list(y_dev.index))  # Dates (index order) are preserved
        self.assertEqual(X_eval.shape, (8, 5))
        self.assertEqual(y_eval.shape, (8,))
        self.assertListEqual(list(X_eval.index), list(y_eval.index))  # Dates (index order) are preserved

        # Test no shuffle split
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(self.preprocessed_data, shuffle=False)
        self.assertEqual(X_dev.shape, (7, 5))
        self.assertEqual(y_dev.shape, (7,))
        self.assertListEqual(list(X_dev.index), list(y_dev.index))  # Dates (index order) are preserved
        self.assertEqual(X_dev.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(X_dev.index[-1], pd.Timestamp('2019-01-11 00:00:00'))
        self.assertEqual(X_eval.shape, (8, 5))
        self.assertEqual(y_eval.shape, (8,))
        self.assertListEqual(list(X_eval.index), list(y_eval.index))  # Dates (index order) are preserved
        self.assertEqual(X_eval.index[0], pd.Timestamp('2019-01-12 00:00:00'))
        self.assertEqual(y_eval.index[-1], pd.Timestamp('2019-01-19 00:00:00'))

    def test_get_optimized_classifier(self):
        # Split dataset for classification
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(self.preprocessed_data, shuffle=SHUFFLE)

        # Grid Search CV with SVMs
        svc = svm.SVC()
        svm_tuned_parameters = [{'kernel': ["rbf", "poly"], 'gamma': [0.1, 1],
                                 'C': [0.1, 1]},
                                {'kernel': ["linear"], 'C': [0.1, 1]}]
        grid_search_cv_svc, svc_training_acc, svc_eval_acc = \
            get_optimized_classifier(svc, svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval, CV_FOLDS)
        pass
