import logging
import numpy as np
import os
import pandas as pd
import sys
from config import load_cryptonalysis_config, CryptonalysisConfig
from preprocessing import get_historical_df, preprocess_dataframe, CRYPTOCURRENCIES, MASTER_DATA_DIR
from training import load_model
from cryptonalysis.utils.data_link import get_crypto_data_for_date
from cryptonalysis.utils.misc_utils import parse_date
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
    logger.info("Preprocessing config:\n" + str(cryptonalysis_config.preprocessing))

    # Preprocessing
    logger.info("Crypto: {}".format(cryptonalysis_config.crypto))
    try:
        data_file = os.path.join(MASTER_DATA_DIR, "new_{}.csv".format(CRYPTOCURRENCIES[cryptonalysis_config.crypto]))
        df = get_historical_df(data_file)

        # Preprocessing parameters
        preprocessing_config = cryptonalysis_config.preprocessing
        predictor_cls = preprocessing_config.predictor_cls
        price_column = preprocessing_config.price_column
        window_size = preprocessing_config.window_size
        normalize = preprocessing_config.normalize
        normalize_by_row = preprocessing_config.normalize_by_row

        # Predictor parameters
        lookahead_days = preprocessing_config.predictor_params['lookahead_days']
        starting_investment = preprocessing_config.predictor_params['starting_investment']
        daily_allowance = preprocessing_config.predictor_params['daily_allowance']

        # Preprocess the data
        preprocessed_data = preprocess_dataframe(df, cryptonalysis_config.crypto, predictor_cls, price_column,
                                                 window_size, normalize, normalize_by_row, starting_date=None,
                                                 ending_date=None, starting_investment=starting_investment,
                                                 daily_allowance=daily_allowance, lookahead_days=lookahead_days)

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


def predict_for_date(cryptonalysis_config, for_date):
    crypto_data = get_crypto_data_for_date(cryptonalysis_config.crypto, for_date,
                                           cryptonalysis_config.preprocessing.window_size)

    # Preprocessing parameters
    preprocessing_config = cryptonalysis_config.preprocessing
    predictor_cls = preprocessing_config.predictor_cls
    price_column = preprocessing_config.price_column
    window_size = preprocessing_config.window_size
    normalize = preprocessing_config.normalize
    normalize_by_row = preprocessing_config.normalize_by_row

    # Predictor parameters
    lookahead_days = preprocessing_config.predictor_params['lookahead_days']
    starting_investment = preprocessing_config.predictor_params['starting_investment']
    daily_allowance = preprocessing_config.predictor_params['daily_allowance']

    # Preprocess the data
    preprocessed_data = preprocess_dataframe(crypto_data, cryptonalysis_config.crypto, predictor_cls, price_column,
                                             window_size, normalize, normalize_by_row, starting_date=None,
                                             ending_date=None, starting_investment=starting_investment,
                                             daily_allowance=daily_allowance, lookahead_days=0)

    return preprocessed_data


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Action not given in args! Must be "simulation" or "prediction".')

    if len(sys.argv) < 3:
        raise ValueError('Config file not given in args!')

    action = sys.argv[1].lower()
    config_file = sys.argv[2]

    config_path = os.path.join(os.path.pardir, os.path.join(os.path.pardir, 'config'))
    config = load_cryptonalysis_config(config_path, config_file)

    if action == 'simulation':
        run_prediction_simulation(config)
    elif action == 'prediction':
        today = parse_date('today')
        predict_for_date(config, today)
    else:
        logger.error("Wrong action \"{}\". Must be \"simulation\" or \"prediction\".".format(action))
