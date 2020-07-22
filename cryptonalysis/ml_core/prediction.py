import logging
import numpy as np
import os
import sys

from cryptonalysis.config import load_cryptonalysis_config, CryptonalysisConfig, PreprocessingConfig, TrainingConfig, \
    DeepLearningConfig
from cryptonalysis.utils.data_link import get_crypto_data_for_date
from cryptonalysis.utils.misc_utils import parse_date
from market import market, MarketParameters
from pandas import DataFrame, Series
from preprocessing import get_historical_df, get_price_list, preprocess_dataframe, preprocess_data_point, \
    CRYPTOCURRENCIES, MASTER_DATA_DIR
from transaction_builders import CryptoPredictor
from training import load_model
from training_dl import load_deep_learning_model
from transaction_builders import get_transaction_type
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


# Logging
logger = logging.getLogger()

# Constants
STARTING_INVESTMENT = 100
DAILY_ALLOWANCE = 5
MARKET_PARAMETERS = {
    'min_transaction_size_crypto': 0.01,
    'min_transaction_size_fiat': 1,
    'max_transaction_size_crypto': 100,
    'max_transaction_size_fiat': 100
}


def get_simple_model_name(model):
    """
    Get a short model name from a model instance.
    :param model: An instance of a model
    :return: A short model name
    :rtype: str
    """
    if isinstance(model, GridSearchCV):
        model_name = model.estimator.__class__.__name__
    else:
        model_name = model.__class__.__name__
    return model_name


def split_dataset(df):
    """
    Split the DataFrame into an attribute vector X and a target vector y for prediction.
    :param df: The DataFrame to split
    :type df: DataFrame
    :return a tuple of attribute and target vectors (X, y)
    :rtype: (DataFrame, DataFrame)
    """
    X = df.iloc[:, :-1]
    y = df['transaction']

    return X, y


def split_dl_datasets(preprocessed_dfs):
    """
    Split each cryptocurrency's preprocessed DataFrame into datasets X and y.
    :param preprocessed_dfs: A dictionary of preprocessed DataFrames per cryptocurrency.
    :type preprocessed_dfs: dict
    :return: A dictionary of DataFrames split intro X and y for each cryptocurrency.
    :rtype: dict
    """
    logger.info("Splitting datasets for classification...")

    split_dfs = {
        crypto: {
            'X': {},
            'y': {}
        }
        for crypto in preprocessed_dfs.keys()
    }

    for (crypto, df) in preprocessed_dfs.iteritems():
        logger.info("*** {} ***".format(crypto))
        X, y = split_dataset(df)
        split_dfs[crypto]['X'] = X
        split_dfs[crypto]['y'] = y

    return split_dfs


def build_deep_learning_datasets(split_dfs, crypto_name):
    """
    Build multi-channel datasets for deep learning model prediction given a dictionary of split DataFrames per crypto
    and a main cryptocurrency for which to build output data.
    The resulting input datasets have a shape of (sample size, price window size, number of cryptos used).
    The resulting output datasets have a shape of (sample size,).
    :param split_dfs: a dictionary of previously split DataFrames (into X, y) per crypto.
    Each cryptocurrency DataFrame is used as a channel in the deep NN
    :type split_dfs: dict
    :param crypto_name: The name of the cryptocurrency whose output will be used to predict in a dl NN
    :type crypto_name: str
    :return: a tuple of (inputs, outputs)
    :rtype: (np.ndarray, np.ndarray)
    """

    inputs = []
    outputs = np.array(split_dfs[crypto_name]['y'])
    for (crypto, dfs) in split_dfs.iteritems():
        inputs.append(dfs['X'])

    inputs = [np.array(i) for i in inputs]
    inputs = np.array(inputs)
    inputs = np.swapaxes(inputs, 0, 1)
    inputs = np.swapaxes(inputs, 1, 2)

    return inputs, outputs


def run_transactions(predictor, y_pred, starting_investment=None, daily_allowance=None):
    """
    Run a list of predicted transactions on a predictor.
    :param predictor: a CryptoPredictor on which to run a simulation for a list of predicted transactions.
    :type predictor: CryptoPredictor
    :param y_pred: predicted transactions
    :rtype: Series
    :return: a dictionary with prediction results
    :param starting_investment: The starting investment to override the one set on the CryptoPredictor instance.
    :type starting_investment: float
    :param daily_allowance: A daily allowance to override the one set on the CryptoPredictor instance.
    :type daily_allowance: float
    :rtype: dict
    """
    predictor.run_transaction_simulation(y_pred.tolist(), starting_investment, daily_allowance)
    cash = predictor.cash
    total_investment = predictor.total_investment
    owned_crypto = predictor.owned_crypto
    return {
        'cash': cash,
        'total_investment': total_investment,
        'owned_crypto': owned_crypto,
        'y_pred': y_pred,
        'predictor_states': predictor.predictor_states
    }


def predict_labeled_data(model, X, y):
    """
    Make a prediction on X and cross-check it against already labeled data y.
    :param model: an instance of a prediction model
    :param X: attributes for prediction
    :type X: DataFrame
    :param y: labeled ground-truth data
    :type y: Series
    :return: a tuple containing the newly predicted data and the accuracy
    :rtype: tuple
    """
    logger.info("Evaluation results for {} model:".format(get_simple_model_name(model)))
    preds = [0 if y_i < 0.5 else 1 for y_i in model.predict(X)]
    y_true, y_pred = y, Series(preds, y.index)
    logger.info('\n' + classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    logger.info("Evaluation accuracy ({}): {}\n".format(get_simple_model_name(model), accuracy))

    return y_pred, accuracy


def run_prediction_simulation(cryptonalysis_config, master_data_dir=None, scaler_dir=None, model_dir=None, models=None,
                              market_parameters=None, starting_investment=None, daily_allowance=None):
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
    :param market_parameters: Overridden market parameters
    :type market_parameters: MarketParameters
    :param starting_investment: The starting investment to override the one set on the CryptoPredictor instance.
    :type starting_investment: float
    :param daily_allowance: A daily allowance to override the one set on the CryptoPredictor instance.
    :type daily_allowance: float
    :return A tuple containing a dictionary with results of the simulation and the ground truth.
    The results dictionary contains the following data:
    {
        'SVC': {
            'cash': svc_cash,
            'total_investment': svc_total_investment,
            'owned_crypto': svc_owned_crypto,
            'y_pred': y_pred_svc,
            'acc': accuracy_svc,
            'predictor_states': predictor states DataFrame
        },
        'MLPClassifier': {
            'cash': mlp_cash,
            'total_investment': mlp_total_investment,
            'owned_crypto': mlp_owned_crypto,
            'y_pred': y_pred_mlp,
            'acc': accuracy_mlp,
            'predictor_states': predictor states DataFrame
        }
    }
    :rtype: (dict, Series)
    """
    logger.info("Running prediction simulation...")
    logger.info("Crypto: {}".format(cryptonalysis_config.crypto))
    logger.info("Preprocessing config:\n" + str(cryptonalysis_config.preprocessing))

    master_data_dir_to_use = master_data_dir or MASTER_DATA_DIR
    market_to_use = market_parameters or market

    # Preprocessing for ground-truth transactions
    try:
        df = None
        preprocessed_dfs = None
        preprocessed_data = None
        if cryptonalysis_config.deep_learning:  # Deep learning path
            preprocessed_dfs = {}
            for crypto in CRYPTOCURRENCIES:
                data_file = os.path.join(master_data_dir_to_use, "new_{}.csv".format(CRYPTOCURRENCIES[crypto]))
                dataframe = get_historical_df(data_file)
                preprocessed_data = preprocess_dataframe(dataframe, crypto, cryptonalysis_config.preprocessing,
                                                         predicting=True, scaler_dir=scaler_dir)
                preprocessed_dfs[crypto] = preprocessed_data
                if crypto == cryptonalysis_config.crypto:
                    df = dataframe
        else:  # Regular training path
            data_file = os.path.join(master_data_dir_to_use,
                                     "new_{}.csv".format(CRYPTOCURRENCIES[cryptonalysis_config.crypto]))
            df = get_historical_df(data_file)
            preprocessed_data = preprocess_dataframe(df, cryptonalysis_config.crypto,
                                                     cryptonalysis_config.preprocessing, predicting=True,
                                                     scaler_dir=scaler_dir)
    except Exception as e:
        logger.error("Error in preprocessing for prediction!", exc_info=e)
        raise e

    # Transaction simulation for model predictions and ROI
    logger.info("Training config:\n" + str(cryptonalysis_config.training))
    try:
        # Preprocessing parameters
        preprocessing_config = cryptonalysis_config.preprocessing
        predictor_cls = preprocessing_config.predictor_cls
        starting_date = preprocessing_config.start_date
        ending_date = preprocessing_config.end_date
        price_column = preprocessing_config.price_column
        window_size = preprocessing_config.window_size

        # Predictor parameters
        lookahead_days = preprocessing_config.predictor_params['lookahead_days']

        prices = get_price_list(df, price_column)
        predictor = predictor_cls(market_to_use, prices, window_size, cryptonalysis_config.crypto, starting_date,
                                  ending_date, preprocessing_config.predictor_params['starting_investment'],
                                  preprocessing_config.predictor_params['daily_allowance'], lookahead_days)

        # Get model(s)
        if cryptonalysis_config.deep_learning:  # Deep learning path
            # Split the datasets for classification
            split_dfs = split_dl_datasets(preprocessed_dfs)
            X, y = build_deep_learning_datasets(split_dfs, cryptonalysis_config.crypto)

            if models:
                lstm_model = models['LSTM']['classifier']
            else:
                lstm_model = load_deep_learning_model(cryptonalysis_config.crypto, preprocessing_config,
                                                      cryptonalysis_config.training, cryptonalysis_config.deep_learning,
                                                      model_dir)

            models = [lstm_model]
        else:  # Regular training path
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

            models = [svc_model, mlp_model]

        # Set index on output
        if hasattr(X, 'index'):
            index = X.index
        else:
            index = preprocessed_dfs[cryptonalysis_config.crypto].index
        y_true = Series(y, index) if isinstance(y, np.ndarray) else y

        # Do the actual predictions
        results = {}
        for model in models:
            model_name = get_simple_model_name(model)

            y_pred, acc = predict_labeled_data(model, X, y_true)

            # Transaction simulation
            transactions_results = run_transactions(predictor, y_pred, starting_investment, daily_allowance)
            results[model_name] = transactions_results
            results[model_name]['acc'] = acc

        logger.info("Prediction simulation complete!\n")
        return results, y_true
    except Exception as e:
        logger.error("Error in prediction simulation!", exc_info=e)
        raise e


def run_multiple_simulations(conf_path, market_parameters=None, starting_investment=None, daily_allowance=None):
    print("HELLO")
    logger.info("Running multiple simulations...")
    config_files = filter(lambda c: c.startswith('training_best') or c.startswith('training_dl_best'),
                          os.listdir(conf_path))
    max_roi = {}
    max_acc = {}
    for conf_file in config_files:
        logger.info("Running simulation for config file '{}'...".format(conf_file))
        conf = load_cryptonalysis_config(conf_path, conf_file)
        results, y_true = run_prediction_simulation(conf, market_parameters=market_parameters,
                                                    starting_investment=starting_investment,
                                                    daily_allowance=daily_allowance)
        yield conf_file, conf, results, y_true  # For external use

        for model in results.keys():
            if model not in max_roi:
                max_roi[model] = {}
                max_roi[model]['roi'] = -sys.maxint
            if model not in max_acc:
                max_acc[model] = {}
                max_acc[model]['acc'] = 0

            roi = results[model]['cash'] - results[model]['total_investment']
            roi_perc = (roi / results[model]['total_investment']) * 100
            acc = results[model]['acc']
            if roi > max_roi[model]['roi']:
                max_roi[model]['roi'] = roi
                max_roi[model]['roi_perc'] = roi_perc
                max_roi[model]['acc'] = acc
                max_roi[model]['conf'] = conf_file
            if acc > max_acc[model]['acc']:
                max_acc[model]['roi'] = roi
                max_acc[model]['roi_perc'] = roi_perc
                max_acc[model]['acc'] = acc
                max_acc[model]['conf'] = conf_file

    for model in max_roi.keys():
        logger.info("Max roi {}: ${}, roi %: {}%, acc: {}, conf: {}".format(model,
                                                                            round(max_roi[model]['roi'], 2),
                                                                            round(max_roi[model]['roi_perc'], 4),
                                                                            round(max_roi[model]['acc'], 2),
                                                                            max_roi[model]['conf']))
        logger.info("Max acc {}: {}, roi: ${} roi %: {}%, conf: {}\n".format(model,
                                                                             round(max_acc[model]['acc'], 2),
                                                                             round(max_acc[model]['roi'], 2),
                                                                             round(max_acc[model]['roi_perc'], 4),
                                                                             max_acc[model]['conf']))


def predict_for_date(crypto_name, preprocessing_config, training_config, deep_learning_config=None, for_date=None,
                     scaler_dir=None, model_dir=None, models=None):
    """
    Predict a transaction (BUY/SELL) for a given date with given configuration objects.
    The data for the given date is extracted from a remote location and preprocessed given the specific configurations.
    A presaved model is loaded with the given configurations to make the prediction.
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.)
    :type crypto_name: str
    :param preprocessing_config: The preprocessing configuration object
    :type preprocessing_config: PreprocessingConfig
    :param training_config: The training configuration object
    :type training_config: TrainingConfig
    :param deep_learning_config: (Default None) The training DL configuration object. If given, a DL model will be
    loaded and used for prediction, otherwise a regular training model will be used.
    :type deep_learning_config: DeepLearningConfig or None
    :param for_date: (Default today) The date for which to make a transaction prediction.
    :type for_date: date
    :param scaler_dir: (Default None) The (overridden) directory where the scaler should be loaded. If None, the default
    `PREPROCESSED_DATA_DIR` is used.
    :type scaler_dir: str
    :param model_dir: (Default None) The (overridden) directory where the model should be loaded from.
    If None, the default `MODELS_DIR` is used.
    :type model_dir: str
    :param models: (Default None) A dictionary with the trained classifiers to use for prediction. If None, the models
    will be loaded from the models directory. Example:
    {
        'SVC': {
            'classifier': SVC
        },
        'MLPClassifier': {
            'classifier': MLPClassifier
        }
    }
    :type models: dict
    :return A dictionary of predictions per classification type. For example:
    {
        'SVC': 'BUY',
        'MLPClassifier': 'SELL',
        'LSTM': None
    }
    or
    {
        'SVC': None,
        'MLPClassifier': None,
        'LSTM': 'BUY'
    }
    :rtype: dict
    """
    logging.info("Obtaining crypto predictions for {}...".format(for_date))

    if deep_learning_config:  # Deep learning path
        # Preprocess the data
        preprocessed_dfs = {}
        for crypto in CRYPTOCURRENCIES:
            crypto_data = get_crypto_data_for_date(crypto, for_date, preprocessing_config.window_size)
            preprocessed_data = preprocess_data_point(crypto_data, crypto, preprocessing_config, scaler_dir)
            preprocessed_dfs[crypto] = preprocessed_data

        # Split the datasets for classification
        split_dfs = split_dl_datasets(preprocessed_dfs)
        X, y = build_deep_learning_datasets(split_dfs, crypto_name)

        if models:
            lstm_model = models['LSTM']['classifier']
        else:
            lstm_model = load_deep_learning_model(crypto_name, preprocessing_config, training_config,
                                                  deep_learning_config, model_dir)

        # Prediction
        pred = 0 if lstm_model.predict(X) < 0.5 else 1
        y_pred_lstm = Series([pred], split_dfs[crypto_name]['y'].index)

        return {
            'LSTM': y_pred_lstm
        }
    else:  # Regular training path
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


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Action not given in args! Must be "simulation" or "prediction".')

    action = sys.argv[1].lower()
    config_path = 'config'

    market_params = MarketParameters(
        min_transaction_size_crypto=MARKET_PARAMETERS['min_transaction_size_crypto'],
        min_transaction_size_fiat=MARKET_PARAMETERS['min_transaction_size_fiat'],
        max_transaction_size_crypto=MARKET_PARAMETERS['max_transaction_size_crypto'],
        max_transaction_size_fiat=MARKET_PARAMETERS['max_transaction_size_fiat'])

    if action == 'multi_simulations':
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Necessary for initial logs
        for _, _, _, _ in run_multiple_simulations(config_path, market_parameters=market_params,
                                                   starting_investment=STARTING_INVESTMENT,
                                                   daily_allowance=DAILY_ALLOWANCE):
            pass
    else:
        if len(sys.argv) < 3:
            raise ValueError('Config file not given in args!')
        config_file = sys.argv[2]
        config = load_cryptonalysis_config(config_path, config_file)

        if action == 'simulation':
            run_prediction_simulation(config, market_parameters=market_params,
                                      starting_investment=STARTING_INVESTMENT, daily_allowance=DAILY_ALLOWANCE)
        elif action == 'prediction':
            today = parse_date('today')
            predictions = predict_for_date(config.crypto, config.preprocessing, config.training, config.deep_learning,
                                           today)
            logger.info('Predictions:')
            for k, v in predictions.items():
                print "{}: {}".format(k, get_transaction_type(v.iloc[0]))
        else:
            logger.error("Wrong action \"{}\". Must be \"simulation\" or \"prediction\".".format(action))
