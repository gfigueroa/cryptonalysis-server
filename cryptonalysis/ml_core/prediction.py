import logging
import numpy as np
import os
import sys
from cryptonalysis.config import load_cryptonalysis_config, CryptonalysisConfig
from cryptonalysis.ml_core.transaction_builders import CryptoPredictor
from cryptonalysis.utils.data_link import get_crypto_data_for_date
from cryptonalysis.utils.misc_utils import parse_date
from market import market
from pandas import DataFrame
from preprocessing import get_historical_df, get_price_list, preprocess_dataframe, preprocess_data_point, \
    CRYPTOCURRENCIES, MASTER_DATA_DIR

from training import load_model
from transaction_builders import get_transaction_type
from sklearn.metrics import classification_report, accuracy_score


# Logging
logger = logging.getLogger()

# Constants
STARTING_INVESTMENT = None
DAILY_ALLOWANCE = None


def split_dataset(df):
    """
    Split the DataFrame into an attribute vector X and a target vector y.
    :param df: The DataFrame to split
    :type df: DataFrame
    :return a tuple of attribute and target vectors (X, y)
    :rtype: (DataFrame, DataFrame)
    """
    X = df.iloc[:, :-1]
    y = df['transaction']

    return X, y


def run_prediction_simulation(cryptonalysis_config, master_data_dir=None, scaler_dir=None, model_dir=None, models=None):
    """
    Run a prediction simulation with unseen data.
    :param cryptonalysis_config
    :type cryptonalysis_config: CryptonalysisConfig
    :param master_data_dir: (Default None) The (overridden) directory where the master data is located. If None, the
    default `MASTER_DATA_DIR` is used.
    :type master_data_dir: str
    :param scaler_dir: (Default None) The (overridden) directory where the scaler is located. If None, the default
    `SCALER_DIR` is used.
    :type scaler_dir: str
    :param model_dir: (Default None) The (overridden) directory where the trained models are located. If None, the
    default `MODEL_DIR` is used.
    :type model_dir: str
    :param models: (Default None) A dictionary with the trained classifiers to use for prediction. If None, the models
    will be loaded from the models directory.
    :type models: dict
    :return The resulting CryptoPredictor instance after running the simulation
    :rtype: CryptoPredictor
    """
    logger.info("Running prediction simulation...")
    logger.info("Preprocessing config:\n" + str(cryptonalysis_config.preprocessing))

    # Preprocessing for ground-truth transactions
    logger.info("Crypto: {}".format(cryptonalysis_config.crypto))
    try:
        master_data_dir_to_use = master_data_dir or MASTER_DATA_DIR
        data_file = os.path.join(master_data_dir_to_use, "new_{}.csv".format(CRYPTOCURRENCIES[cryptonalysis_config.crypto]))
        df = get_historical_df(data_file)

        # Preprocess the data
        preprocessed_data = preprocess_dataframe(df, cryptonalysis_config.crypto, cryptonalysis_config.preprocessing,
                                                 predicting=True, scaler_dir=scaler_dir)
    except Exception as e:
        logger.error("Error in preprocessing for prediction!")
        logger.error(e.message)
        raise e

    # Transaction simulation for model predictions and ROI
    logger.info("Training config:\n" + str(cryptonalysis_config.training))
    try:
        # Split dataset for classification
        X, y = split_dataset(preprocessed_data)

        if models:
            svc_model = models['SVC']['classifier']
            mlp_model = models['MLPClassifier']['classifier']
        else:
            svc_model = load_model('SVC', cryptonalysis_config.crypto, cryptonalysis_config.preprocessing,
                                   cryptonalysis_config.training, model_dir=model_dir)
            mlp_model = load_model('MLPClassifier', cryptonalysis_config.crypto, cryptonalysis_config.preprocessing,
                                   cryptonalysis_config.training, model_dir=model_dir)

        # Evaluation dataset
        logger.info("Evaluation results for SVC model:")
        y_true, y_pred_svc = y, svc_model.predict(X)
        logger.info('\n' + classification_report(y_true, y_pred_svc))
        accuracy = accuracy_score(y_true, y_pred_svc)
        logger.info("Evaluation accuracy (SVC): {}\n".format(accuracy))

        logger.info("Evaluation results for MLPClassifier model:")
        y_true, y_pred_mlp = y, mlp_model.predict(X)
        logger.info('\n' + classification_report(y_true, y_pred_mlp))
        accuracy = accuracy_score(y_true, y_pred_mlp)
        logger.info("Evaluation accuracy (MLPClassifier): {}\n".format(accuracy))

        # Run transactions to calculate ROI
        # Preprocessing parameters
        preprocessing_config = cryptonalysis_config.preprocessing
        predictor_cls = preprocessing_config.predictor_cls
        starting_date = preprocessing_config.start_date
        ending_date = preprocessing_config.end_date
        price_column = preprocessing_config.price_column
        window_size = preprocessing_config.window_size

        # Predictor parameters
        lookahead_days = preprocessing_config.predictor_params['lookahead_days']
        starting_investment = preprocessing_config.predictor_params['starting_investment']
        daily_allowance = preprocessing_config.predictor_params['daily_allowance']

        prices = get_price_list(df, price_column)
        predictor = predictor_cls(market, prices, window_size, cryptonalysis_config.crypto, starting_date, ending_date,
                                  starting_investment, daily_allowance, lookahead_days)

        logger.info("Transaction simulation for SVC...")
        predictor.run_transaction_simulation(y_pred_svc.tolist(), STARTING_INVESTMENT, DAILY_ALLOWANCE)
        logger.info("Transaction simulation for MLPClassifier...")
        predictor.run_transaction_simulation(y_pred_mlp.tolist(), STARTING_INVESTMENT, DAILY_ALLOWANCE)

        logger.info("Prediction simulation complete!\n")
        
        return predictor
    except Exception as e:
        logger.error("Error in prediction simulation!")
        logger.error(e.message)
        raise e


def predict_for_date(crypto_name, preprocessing_config, training_config, for_date, scaler_dir=None, model_dir=None):
    """
    Predict a transaction (BUY/SELL) for a given date with a given CryptonalysisConfig object.
    The data for the given date is extracted from a remote location and preprocessed given the specific configuration.
    A presaved model is loaded with the given configuration to make the prediction.
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.)
    :type crypto_name: str
    :param preprocessing_config: The preprocessing configuration object
    :type preprocessing_config: PreprocessingConfig
    :param training_config: The training configuration object
    :type training_config: TrainingConfig
    :param for_date: The date for which to make a transaction prediction.
    :type for_date: date
    :param scaler_dir: (Default None) The (overridden) directory where the scaler should be loaded. If None, the default
    `PREPROCESSED_DATA_DIR` is used.
    :type scaler_dir: str
    :param model_dir: (Default None) The (overridden) directory where the model should be loaded from.
    If None, the default `MODELS_DIR` is used.
    :type model_dir: str
    :return A dictionary of predictions per classification type. For example:
    {
        SVC: BUY,
        MLPClassifier: SELL
    }
    :rtype: dict
    """
    logging.info("Obtaining crypto predictions for {}...".format(for_date))
    crypto_data = get_crypto_data_for_date(crypto_name, for_date, preprocessing_config.window_size)

    # Preprocess the data
    preprocessed_data = preprocess_data_point(crypto_data, crypto_name, preprocessing_config, scaler_dir)

    # Split dataset for classification
    X, y = split_dataset(preprocessed_data)

    svc_model = load_model('SVC', crypto_name, preprocessing_config, training_config, model_dir)
    mlp_model = load_model('MLPClassifier', crypto_name, preprocessing_config, training_config, model_dir)

    # Prediction
    y_pred_svc = get_transaction_type(svc_model.predict(X)[0])
    y_pred_mlp = get_transaction_type(mlp_model.predict(X)[0])

    return {
        'SVC': y_pred_svc,
        'MLPClassifier': y_pred_mlp
    }


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Action not given in args! Must be "simulation" or "prediction".')

    if len(sys.argv) < 3:
        raise ValueError('Config file not given in args!')

    action = sys.argv[1].lower()
    config_file = sys.argv[2]

    config_path = 'config'
    config = load_cryptonalysis_config(config_path, config_file)

    if action == 'simulation':
        run_prediction_simulation(config)
    elif action == 'prediction':
        today = parse_date('today')
        predictions = predict_for_date(config.crypto, config.preprocessing, config.training, today)
        logger.info("Predictions:\n{}".format(predictions))
    else:
        logger.error("Wrong action \"{}\". Must be \"simulation\" or \"prediction\".".format(action))
