import logging
import numpy as np
import os
import pandas as pd
import sys
from config import load_cryptonalysis_config, PreprocessingConfig, TrainingConfig, CryptonalysisConfig
from datetime import datetime
from preprocessing import run_preprocessing_pipeline
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier


# Logging
logger = logging.getLogger()

RESULTS_DIR = os.path.join(os.path.pardir, 'results')


def run_prediction_simulation_pipeline(preprocessed_df, training_config):
    """
    Run the prediction simulation pipeline.
    :param preprocessed_df: The preprocessed DataFrame ready for the classification pipeline
    :param training_config
    :type training_config: TrainingConfig
    """

    logger.info("Running prediction simulation pipeline...")
    logger.info("Training config:\n" + str(training_config))

    # Split dataset for classification
    X, y, X_training, y_training, X_testing, y_testing, X_dev, y_dev, X_eval, y_eval = \
        split_datasets(preprocessed_df, training_config.shuffle_data, training_config.training_size,
                       training_config.dev_size)

    # Grid Search CV with SVMs
    svc = svm.SVC()
    svm_tuned_parameters = [{'kernel': ["rbf", "poly"], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 1],
                             'C': [0.01, 0.1, 1, 10]},
                            {'kernel': ["linear"], 'C': [0.01, 0.1, 1, 10]}]
    grid_search_cv_svc, svc_training_acc, svc_eval_acc = \
        get_optimized_classifier(svc, svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval, training_config.cv_folds)

    # Grid Search CV with NNs
    mlp = MLPClassifier()
    mlp_tuned_parameters = {
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'hidden_layer_sizes': [(10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50)],
        'alpha': [0.1, 1, 10],
        'activation': ["identity", "logistic", "tanh", "relu"]
    }
    grid_search_cv_mlp, mlp_training_acc, mlp_eval_acc = \
        get_optimized_classifier(mlp, mlp_tuned_parameters, X_dev, y_dev, X_eval, y_eval)

    logger.info("Classification pipeline complete!\n")

    trained_classifiers = {
        'svc': {
            'classifier': grid_search_cv_svc,
            'training_acc': svc_training_acc,
            'eval_acc': svc_eval_acc
        },
        'mlp': {
            'classifier': grid_search_cv_mlp,
            'training_acc': mlp_training_acc,
            'eval_acc': mlp_eval_acc
        }
    }

    return trained_classifiers


def run_prediction_simulation(cryptonalysis_config):
    """
    Run a prediction simulation with unused data.
    :param cryptonalysis_config
    :type cryptonalysis_config: CryptonalysisConfig
    """

    # Preprocessing
    logger.info("Crypto: {}".format(cryptonalysis_config.crypto))
    try:
        preprocessed_data = run_preprocessing_pipeline(cryptonalysis_config.crypto, cryptonalysis_config.preprocessing)
    except Exception as e:
        logger.error("Error in preprocessing pipeline! Skipping...")
        logger.error(e.message)
        raise e

    # Simulation
    try:
        run_prediction_simulation_pipeline(preprocessed_data, cryptonalysis_config.training)
    except Exception as e:
        logger.error("Error in training pipeline! Skipping...")
        logger.error(e.message)
        raise e


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Config file not given in args!')

    config_file = sys.argv[1]

    config_path = os.path.join(os.path.pardir, os.path.join(os.path.pardir, 'config'))
    config = load_cryptonalysis_config(config_path, config_file)

    run_prediction_simulation(config)
