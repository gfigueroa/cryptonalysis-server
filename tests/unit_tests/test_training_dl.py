"""
Unit tests for the training_dl.py module.
"""

import numpy
import os
import random
import unittest
from cryptonalysis.config import PreprocessingConfig, TrainingConfig, DeepLearningConfig
from cryptonalysis.ml_core.preprocessing import CRYPTOCURRENCIES, run_preprocessing_pipeline
from cryptonalysis.ml_core.training_dl import get_split_dfs, build_deep_learning_datasets, build_model, \
    save_deep_learning_model, load_deep_learning_model
from cryptonalysis.ml_core.transaction_builders import get_predictor_class_from_name
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

# Deep Learning constants for tests
NEURONS = 10
DEEP_LEARNING_CONFIG = DeepLearningConfig({
    'neurons': NEURONS
})


class TestDeepLearningFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDeepLearningFunctions, self).__init__(*args, **kwargs)

        self.preprocessed_dfs = {
            crypto_name: run_preprocessing_pipeline(crypto_name, PREPROCESSING_CONFIG, save_data=False, save_roi=False,
                                                    master_data_dir=RES_DIR)
            for crypto_name in CRYPTOCURRENCIES
        }

        # Always have the same random arrangements
        random.seed(1)
        numpy.random.seed(1)

    def test_get_split_dfs(self):
        split_dfs = get_split_dfs(self.preprocessed_dfs, TRAINING_CONFIG)

        # Indices are all the same
        dev_indices = []
        X_devs = [df['X_dev'] for df in split_dfs.values()]
        indices = [list(df.index) for df in X_devs]
        for i1, i2 in zip(indices, indices[1:]):
            self.assertListEqual(i1, i2)
        dev_indices += indices

        y_devs = [df['y_dev'] for df in split_dfs.values()]
        indices = [list(df.index) for df in y_devs]
        for i1, i2 in zip(indices, indices[1:]):
            self.assertListEqual(i1, i2)
        dev_indices += indices

        for i1, i2 in zip(dev_indices, dev_indices[1:]):
            self.assertListEqual(i1, i2)

        eval_indices = []
        X_evals = [df['X_eval'] for df in split_dfs.values()]
        indices = [list(df.index) for df in X_evals]
        for i1, i2 in zip(indices, indices[1:]):
            self.assertListEqual(i1, i2)
        eval_indices += indices

        y_evals = [df['y_eval'] for df in split_dfs.values()]
        indices = [list(df.index) for df in y_evals]
        for i1, i2 in zip(indices, indices[1:]):
            self.assertListEqual(i1, i2)
        eval_indices += indices

        for i1, i2 in zip(eval_indices, eval_indices[1:]):
            self.assertListEqual(i1, i2)

    def test_build_deep_learning_datasets(self):
        # Split DFs
        split_dfs = get_split_dfs(self.preprocessed_dfs, TRAINING_CONFIG)

        # Build datasets
        training_inputs, training_outputs, test_inputs, test_outputs = \
            build_deep_learning_datasets(split_dfs, CRYPTO_NAME)
        # Check correct shapes
        self.assertEquals(training_inputs.shape, (7, 5, 4))  # 7 rows, 5 days, 4 cryptos
        self.assertEquals(training_outputs.shape, (7,))
        self.assertEquals(test_inputs.shape, (8, 5, 4))
        self.assertEquals(test_outputs.shape, (8,))

        # Check correct target values
        true_target_training = list(split_dfs[CRYPTO_NAME]['y_dev'])
        new_target_training = list(training_outputs)
        self.assertListEqual(true_target_training, new_target_training)
        true_target_testing = list(split_dfs[CRYPTO_NAME]['y_eval'])
        new_target_testing = list(test_outputs)
        self.assertListEqual(true_target_testing, new_target_testing)

    def test_build_model(self):
        # Split DFs
        split_dfs = get_split_dfs(self.preprocessed_dfs, TRAINING_CONFIG)

        # Build datasets
        training_inputs, training_outputs, test_inputs, test_outputs = \
            build_deep_learning_datasets(split_dfs, CRYPTO_NAME)

        # Build model
        model = build_model(input_shape=(training_inputs.shape[1], training_inputs.shape[2]), output_size=1,
                            neurons=NEURONS)
        self.assertEqual(model.input_shape, (None, 5, 4))

        # Fit model
        model.fit(training_inputs, training_outputs, epochs=10, batch_size=1, verbose=1, shuffle=True)

        # Evaluate model
        metric_values = model.evaluate(test_inputs, test_outputs, verbose=1)
        metrics = {metric_name: metric_value for (metric_name, metric_value) in zip(model.metrics_names, metric_values)}
        self.assertAlmostEqual(metrics['acc'], 0.25, 1)
        self.assertAlmostEqual(metrics['loss'], 0.5, 1)

    def test_load_and_save_model(self):
        # Split DFs
        split_dfs = get_split_dfs(self.preprocessed_dfs, TRAINING_CONFIG)

        # Build datasets
        training_inputs, training_outputs, test_inputs, test_outputs = \
            build_deep_learning_datasets(split_dfs, CRYPTO_NAME)

        # Build model
        model = build_model(input_shape=(training_inputs.shape[1], training_inputs.shape[2]), output_size=1,
                            neurons=NEURONS)

        # Fit model
        model.fit(training_inputs, training_outputs, epochs=10, batch_size=1, verbose=1, shuffle=True)

        # Save model
        save_deep_learning_model(model, CRYPTO_NAME, PREPROCESSING_CONFIG, TRAINING_CONFIG, DEEP_LEARNING_CONFIG,
                                 RES_DIR)
        loaded_model = load_deep_learning_model(CRYPTO_NAME, PREPROCESSING_CONFIG, TRAINING_CONFIG,
                                                DEEP_LEARNING_CONFIG, RES_DIR)
        self.assertTrue(loaded_model is not None)
