"""
Unit tests for the prediction.py module.
"""

import os
import unittest
from cryptonalysis.config import PreprocessingConfig, TrainingConfig
from cryptonalysis.ml_core.prediction import split_dataset, predict_for_date
from cryptonalysis.ml_core.preprocessing import load_preprocessed_data
from cryptonalysis.ml_core.transaction_builders import get_predictor_class_from_name
from cryptonalysis.utils.misc_utils import parse_date
from datetime import date

RES_DIR = os.path.join('tests', 'resources')


# Preprocessing constants for tests
CRYPTO_NAME = 'ETH'
STARTING_DATE = date(2019, 1, 1)
ENDING_DATE = date(2019, 1, 20)
WINDOW_SIZE = 5
NORMALIZE = True
NORMALIZE_BY_ROW = True
STANDARDIZE = False
PRICE_COLUMN = 'Close'
PREDICTOR_CLS_NAME = 'BiffPredictor'
PREDICTOR_CLS = get_predictor_class_from_name(PREDICTOR_CLS_NAME)
PROB_BUY = 1
PROB_SELL = 1
STARTING_INVESTMENT = 100
DAILY_ALLOWANCE = 5
LOOKAHEAD_DAYS = 1
PREPROCESSING_CONFIG = PreprocessingConfig({
    'start_date': STARTING_DATE,
    'end_date': ENDING_DATE,
    'window_size': WINDOW_SIZE,
    'normalize': NORMALIZE,
    'normalize_by_row': NORMALIZE_BY_ROW,
    'standardize': STANDARDIZE,
    'price_column': PRICE_COLUMN,
    'predictor_cls': PREDICTOR_CLS_NAME,
    'predictor_params': {
        'prob_buy': PROB_BUY,
        'prob_sell': PROB_SELL,
        'starting_investment': STARTING_INVESTMENT,
        'daily_allowance': DAILY_ALLOWANCE,
        'lookahead_days': LOOKAHEAD_DAYS
    }
})

# Training constants for tests
SHUFFLE = True
CV_FOLDS = 2  # Too restrictive when y_dev is very small
TRAINING_CONFIG = TrainingConfig({
    'shuffle': SHUFFLE,
    'cv_folds': CV_FOLDS
})

# Prediction constants for tests
FOR_DATE = parse_date('2019-02-01')


class TestPredictionFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPredictionFunctions, self).__init__(*args, **kwargs)

        self.preprocessed_data = \
            load_preprocessed_data(crypto_name=CRYPTO_NAME, preprocessing_config=PREPROCESSING_CONFIG,
                                   preprocessed_data_dir=RES_DIR)

    def test_split_dataset(self):
        columns = ['price 1', 'price 2', 'price 3', 'price 4', 'price 5']
        expected_transactions = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]

        X, y = split_dataset(self.preprocessed_data)
        self.assertEqual(X.shape, (15, 5))
        self.assertListEqual(list(X.columns), columns)
        self.assertEqual(y.shape, (15,))
        self.assertListEqual(list(y.values), expected_transactions)

    def test_predict_for_date(self):
        expected_predictions = {
            'SVC': 'SELL',
            'MLPClassifier': 'BUY'
        }
        predictions = predict_for_date(CRYPTO_NAME, PREPROCESSING_CONFIG, TRAINING_CONFIG, FOR_DATE, RES_DIR, RES_DIR)
        self.assertDictEqual(predictions, expected_predictions)
