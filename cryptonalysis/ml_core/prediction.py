import logging
import numpy as np
import os
import sys
from cryptonalysis.config import load_cryptonalysis_config, CryptonalysisConfig
from cryptonalysis.utils.data_link import get_crypto_data_for_date
from cryptonalysis.utils.misc_utils import parse_date
from market import market
from pandas import DataFrame, Series
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
    :return A dictionary with results of the simulation.
    {
        'predictor': predictor instance,
        'SVC': {
            'cash': svc_cash,
            'total_investment': svc_total_investment,
            'owned_crypto': svc_owned_crypto,
            'y_pred': y_pred_svc
        },
        'MLPClassifier': {
            'cash': mlp_cash,
            'total_investment': mlp_total_investment,
            'owned_crypto': mlp_owned_crypto,
            'y_pred': y_pred_mlp
        }
    }
    :rtype: dict
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
        y_true, y_pred_svc = y, Series(svc_model.predict(X), y.index)
        logger.info('\n' + classification_report(y_true, y_pred_svc))
        accuracy = accuracy_score(y_true, y_pred_svc)
        logger.info("Evaluation accuracy (SVC): {}\n".format(accuracy))

        logger.info("Evaluation results for MLPClassifier model:")
        y_true, y_pred_mlp = y, Series(mlp_model.predict(X), y.index)
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
        svc_cash = predictor.cash
        svc_total_investment = predictor.total_investment
        svc_owned_crypto = predictor.owned_crypto
        logger.info("Transaction simulation for MLPClassifier...")
        predictor.run_transaction_simulation(y_pred_mlp.tolist(), STARTING_INVESTMENT, DAILY_ALLOWANCE)
        mlp_cash = predictor.cash
        mlp_total_investment = predictor.total_investment
        mlp_owned_crypto = predictor.owned_crypto

        logger.info("Prediction simulation complete!\n")

        results = {
            'predictor': predictor,
            'SVC': {
                'cash': svc_cash,
                'total_investment': svc_total_investment,
                'owned_crypto': svc_owned_crypto,
                'y_pred': y_pred_svc
            },
            'MLPClassifier': {
                'cash': mlp_cash,
                'total_investment': mlp_total_investment,
                'owned_crypto': mlp_owned_crypto,
                'y_pred': y_pred_mlp
            }
        }
        
        return results
    except Exception as e:
        logger.error("Error in prediction simulation!")
        logger.error(e.message)
        raise e


def predict_for_date(crypto_name, preprocessing_config, training_config, for_date, scaler_dir=None, model_dir=None,
                     models=None):
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
    :param models: (Default None) A dictionary with the trained classifiers to use for prediction. If None, the models
    will be loaded from the models directory.
    :type models: dict
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

    if models:
        svc_model = models['SVC']['classifier']
        mlp_model = models['MLPClassifier']['classifier']
    else:
        svc_model = load_model('SVC', crypto_name, preprocessing_config, training_config, model_dir)
        mlp_model = load_model('MLPClassifier', crypto_name, preprocessing_config, training_config, model_dir)

    # Prediction
    y_pred_svc = Series(svc_model.predict(X), y.index)
    y_pred_mlp = Series(mlp_model.predict(X), y.index)

    return {
        'SVC': y_pred_svc,
        'MLPClassifier': y_pred_mlp
    }


def run_multiple_simulations(conf_path):
    config_files = filter(lambda c: c.startswith('training_best'), os.listdir(conf_path))
    max_roi_svc = 0
    max_roi_conf_svc = None
    max_roi_perc_svc = 0
    max_roi_perc_conf_svc = None
    max_roi_mlp = 0
    max_roi_conf_mlp = None
    max_roi_perc_mlp = 0
    max_roi_perc_conf_mlp = None
    for conf_file in config_files:
        logger.info("Running simulation for config file '{}'".format(conf_file))
        conf = load_cryptonalysis_config(config_path, conf_file)
        results = run_prediction_simulation(conf)

        svc_roi = results['SVC']['cash'] - results['SVC']['total_investment']
        svc_roi_perc = svc_roi / results['SVC']['total_investment']
        if svc_roi > max_roi_svc:
            max_roi_svc = svc_roi
            max_roi_conf_svc = conf_file
        if svc_roi_perc > max_roi_perc_svc:
            max_roi_perc_svc = svc_roi_perc
            max_roi_perc_conf_svc = conf_file

        mlp_roi = results['MLPClassifier']['cash'] - results['MLPClassifier']['total_investment']
        mlp_roi_perc = mlp_roi / results['MLPClassifier']['total_investment']
        if mlp_roi > max_roi_mlp:
            max_roi_mlp = mlp_roi
            max_roi_conf_mlp = conf_file
        if mlp_roi_perc > max_roi_perc_mlp:
            max_roi_perc_mlp = mlp_roi_perc
            max_roi_perc_conf_mlp = conf_file

    logger.info("Max roi svc: {}, conf: {}".format(max_roi_svc, max_roi_conf_svc))
    logger.info("Max roi % svc: {}, conf: {}".format(max_roi_perc_svc, max_roi_perc_conf_svc))

    logger.info("Max roi mlp: {}, conf: {}".format(max_roi_mlp, max_roi_conf_mlp))
    logger.info("Max roi % mlp: {}, conf: {}".format(max_roi_perc_mlp, max_roi_perc_conf_mlp))


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Action not given in args! Must be "simulation" or "prediction".')

    action = sys.argv[1].lower()
    config_path = 'config'

    if action == 'multi_simulations':
        run_multiple_simulations(config_path)
    else:
        if len(sys.argv) < 3:
            raise ValueError('Config file not given in args!')
        config_file = sys.argv[2]
        config = load_cryptonalysis_config(config_path, config_file)

        if action == 'simulation':
            run_prediction_simulation(config)
        elif action == 'prediction':
            today = parse_date('today')
            predictions = predict_for_date(config.crypto, config.preprocessing, config.training, today)
            logger.info('Predictions:')
            for k, v in predictions.items():
                print "{}: {}".format(k, get_transaction_type(v.iloc[0]))
        else:
            logger.error("Wrong action \"{}\". Must be \"simulation\" or \"prediction\".".format(action))
