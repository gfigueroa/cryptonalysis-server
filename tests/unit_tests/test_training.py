"""
Unit tests for the training.py module.
"""

import numpy
import os
import pandas as pd
import random
import unittest
from cryptonalysis.config import PreprocessingConfig, TrainingConfig
from cryptonalysis.ml_core.preprocessing import load_preprocessed_data
from cryptonalysis.ml_core.training import split_datasets, get_optimized_classifier, save_model, load_model
from cryptonalysis.ml_core.transaction_builders import get_predictor_class_from_name
from datetime import date
from sklearn import svm
from sklearn.neural_network import MLPClassifier

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


class TestTrainingFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTrainingFunctions, self).__init__(*args, **kwargs)

        self.preprocessed_data = \
            load_preprocessed_data(crypto_name=CRYPTO_NAME, preprocessing_config=PREPROCESSING_CONFIG,
                                   preprocessed_data_dir=RES_DIR)

        self.svm_tuned_parameters = [{'kernel': ["rbf", "poly"], 'gamma': [0.1, 1],
                                     'C': [0.1, 1]},
                                     {'kernel': ["linear"], 'C': [0.1, 1]}]
        self.mlp_tuned_parameters = {
            'learning_rate': ["constant", "invscaling"],
            'hidden_layer_sizes': [(10, 10), (20, 20)],
            'alpha': [0.1, 1],
            'activation': ["identity", "logistic"]
        }

        # Always have the same random arrangements
        random.seed(1)
        numpy.random.seed(1)

    def test_split_datasets(self):
        # Test default shuffle split
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(self.preprocessed_data, shuffle=True)
        self.assertEqual(X_dev.shape, (7, 5))
        self.assertEqual(y_dev.shape, (7,))
        self.assertListEqual(list(X_dev.index), list(y_dev.index))  # Dates (index order) are preserved
        self.assertEqual(X_eval.shape, (8, 5))
        self.assertEqual(y_eval.shape, (8,))
        self.assertListEqual(list(X_eval.index), list(y_eval.index))  # Dates (index order) are preserved
        intersection = set(X_dev.index).intersection(X_eval.index)
        self.assertTrue(len(intersection) == 0)  # Index intersection is empty

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
        intersection = set(X_dev.index).intersection(X_eval.index)
        self.assertTrue(len(intersection) == 0)  # Index intersection is empty

    def test_get_optimized_classifier_with_shuffle(self):
        # Split dataset for classification
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(self.preprocessed_data, shuffle=SHUFFLE)

        # Grid Search CV with SVMs
        svc = svm.SVC()

        # Check y_dev contains both classes at least K times
        if len(y_dev[y_dev == 0]) < CV_FOLDS or len(y_dev[y_dev == 1]) < CV_FOLDS:
            with self.assertRaises(ValueError):
                get_optimized_classifier(svc, self.svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval, CV_FOLDS)
        else:
            grid_search_cv_svc, svc_training_acc, svc_eval_acc = \
                get_optimized_classifier(svc, self.svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval, CV_FOLDS)
            self.assertEqual(grid_search_cv_svc.best_score_, svc_training_acc)

        # Grid Search CV with NNs
        mlp = MLPClassifier()
        grid_search_cv_mlp, mlp_training_acc, mlp_eval_acc = \
            get_optimized_classifier(mlp, self.mlp_tuned_parameters, X_dev, y_dev, X_eval, y_eval)
        self.assertEqual(grid_search_cv_mlp.best_score_, mlp_training_acc)

    def test_get_optimized_classifier_without_shuffle(self):
        # Split dataset for classification
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(self.preprocessed_data, shuffle=False)

        # Grid Search CV with SVMs
        svc = svm.SVC()
        grid_search_cv_svc, svc_training_acc, svc_eval_acc = \
            get_optimized_classifier(svc, self.svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval, CV_FOLDS)
        self.assertAlmostEquals(svc_training_acc, 0.71429, 5)
        self.assertEqual(svc_eval_acc, 0.5)

        # Grid Search CV with NNs
        mlp = MLPClassifier()
        grid_search_cv_mlp, mlp_training_acc, mlp_eval_acc = \
            get_optimized_classifier(mlp, self.mlp_tuned_parameters, X_dev, y_dev, X_eval, y_eval)
        self.assertAlmostEquals(mlp_training_acc, 0.71429, 5)
        self.assertEqual(mlp_eval_acc, 0.25)

    def test_save_and_load_model(self):
        # Split dataset for classification
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(self.preprocessed_data, shuffle=SHUFFLE)

        # Grid Search CV with NNs
        mlp = MLPClassifier()
        grid_search_cv_mlp, mlp_training_acc, mlp_eval_acc = \
            get_optimized_classifier(mlp, self.mlp_tuned_parameters, X_dev, y_dev, X_eval, y_eval)
        save_model(grid_search_cv_mlp, CRYPTO_NAME, PREPROCESSING_CONFIG, TRAINING_CONFIG, RES_DIR)

        saved_model = load_model(grid_search_cv_mlp.estimator.__class__.__name__, CRYPTO_NAME, PREPROCESSING_CONFIG,
                                 TRAINING_CONFIG, RES_DIR)
        self.assertEqual(grid_search_cv_mlp.best_score_, saved_model.best_score_)
        self.assertEqual(grid_search_cv_mlp.best_index_, saved_model.best_index_)
        self.assertDictEqual(grid_search_cv_mlp.best_params_, saved_model.best_params_)

        # Grid Search CV with SVMs
        svc = svm.SVC()
        grid_search_cv_svc, svc_training_acc, svc_eval_acc = \
            get_optimized_classifier(svc, self.svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval, CV_FOLDS)
        save_model(grid_search_cv_svc, CRYPTO_NAME, PREPROCESSING_CONFIG, TRAINING_CONFIG, RES_DIR)

        saved_model = load_model(grid_search_cv_svc.estimator.__class__.__name__, CRYPTO_NAME, PREPROCESSING_CONFIG,
                                 TRAINING_CONFIG, RES_DIR)
        self.assertEqual(grid_search_cv_svc.best_score_, saved_model.best_score_)
        self.assertEqual(grid_search_cv_svc.best_index_, saved_model.best_index_)
        self.assertDictEqual(grid_search_cv_svc.best_params_, saved_model.best_params_)
