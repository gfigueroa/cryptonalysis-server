import logging
import numpy as np
import os
import pandas as pd
import sys
from config import load_cryptonalysis_config, CryptonalysisConfig
from preprocessing import run_preprocessing_pipeline
from training import load_model
from sklearn.metrics import classification_report, accuracy_score


# Logging
logger = logging.getLogger()

RESULTS_DIR = os.path.join(os.path.pardir, 'results')


def split_dataset(df):
    """
    Split the DataFrame into an attribute vector X and a target vector y.
    :param df: The DataFrame to split
    :type df: pd.DataFrame
    :return: a tuple of (X, y)
    :rtype: (pd.DataFrame, pd.DataFrame)
    """
    X = df.iloc[:, :-1]
    y = df['transaction']

    return X, y


def run_prediction_simulation(cryptonalysis_config):
    """
    Run a prediction simulation with unused data.
    :param cryptonalysis_config
    :type cryptonalysis_config: CryptonalysisConfig
    """
    logger.info("Running prediction simulation...")

    # Preprocessing
    logger.info("Crypto: {}".format(cryptonalysis_config.crypto))
    try:
        preprocessed_data = run_preprocessing_pipeline(cryptonalysis_config.crypto, cryptonalysis_config.preprocessing,
                                                       cryptonalysis_config.save_preprocessing_data,
                                                       cryptonalysis_config.save_preprocessing_roi,
                                                       simulation=True)
    except Exception as e:
        logger.error("Error in preprocessing pipeline! Skipping...")
        logger.error(e.message)
        raise e

    # Simulation
    logger.info("Training config:\n" + str(cryptonalysis_config.training))
    try:
        # Split dataset for classification
        X, y = split_dataset(preprocessed_data)

        svc_model = load_model('svc', cryptonalysis_config.crypto, cryptonalysis_config.preprocessing,
                               cryptonalysis_config.training)
        mlp_model = load_model('mlp', cryptonalysis_config.crypto, cryptonalysis_config.preprocessing,
                               cryptonalysis_config.training)

        # Evaluation dataset
        logger.info("Evaluation results for SVC model:")
        y_true, y_pred = y, svc_model.predict(X)
        logger.info('\n' + classification_report(y_true, y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        logger.info("Evaluation accuracy (SVC): {}\n".format(accuracy))

        logger.info("Evaluation results for MLP model:")
        y_true, y_pred = y, mlp_model.predict(X)
        logger.info('\n' + classification_report(y_true, y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        logger.info("Evaluation accuracy (MLP): {}\n".format(accuracy))

        logger.info("Prediction simulation complete!\n")
    except Exception as e:
        logger.error("Error in prediction simulation! Skipping...")
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
