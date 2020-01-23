# Methods for deep learning

import logging
import numpy as np
import os
import sys
from cryptonalysis.config import load_cryptonalysis_config_grid, PreprocessingConfig,  TrainingConfig, \
    DeepLearningConfig, CryptonalysisConfigGrid
from datetime import datetime
from preprocessing import run_preprocessing_pipeline, CRYPTOCURRENCIES
from training import split_datasets
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Logging
logger = logging.getLogger()

RESULTS_DIR = os.path.join('cryptonalysis', 'results_dl')
MODELS_DIR = os.path.join('cryptonalysis', 'models_dl')


def get_split_dfs(preprocessed_dfs, training_config):
    """
    Split each cryptocurrency's preprocessed DataFrame into development and evaluation datasets X and y.
    :param preprocessed_dfs: A dictionary of preprocessed DataFrames per cryptocurrency.
    :type preprocessed_dfs: dict
    :param training_config
    :type training_config: TrainingConfig
    :return: A dictionary of DataFrames split intro development and evaluation for each cryptocurrency.
    :rtype: dict
    """
    logger.info("Splitting datasets for classification...")
    logger.info("Training config:\n" + str(training_config))

    split_dfs = {
        crypto: {
            'X_dev': {},
            'y_dev': {},
            'X_eval': {},
            'y_eval': {},
        }
        for crypto in preprocessed_dfs.keys()
    }
    shuffled_indices = None  # When shuffling, used to shuffle all dataframes with the same arrangement
    for (crypto, df) in preprocessed_dfs.iteritems():
        logger.info("*** {} ***".format(crypto))
        X, _, X_training, _, X_testing, _, X_dev, y_dev, X_eval, y_eval = split_datasets(
            df, training_config.shuffle_data, training_config.training_size, training_config.dev_size, shuffled_indices)

        # Get shuffled indices once
        if training_config.shuffle_data and not shuffled_indices:
            shuffled_indices = {
                'cv': X.index,
                'training': X_training.index,
                'testing': X_testing.index,
                'dev': X_dev.index,
                'eval': X_eval.index
            }

        split_dfs[crypto]['X_dev'] = X_dev
        split_dfs[crypto]['y_dev'] = y_dev
        split_dfs[crypto]['X_eval'] = X_eval
        split_dfs[crypto]['y_eval'] = y_eval

    return split_dfs


def build_deep_learning_datasets(split_dfs, crypto_name):
    """
    Build multi-channel datasets for deep learning models given a dictionary of split DataFrames per crypto and a main
    cryptocurrency for which to build output data.
    The resulting input datasets have a shape of (sample size, price window size, number of cryptos used).
    The resulting output datasets have a shape of (sample size,).
    :param split_dfs: a dictionary of previously split DataFrames (into X_dev, y_dev, X_eval, y_eval) per crypto.
    Each cryptocurrency DataFrame is used as a channel in the deep NN
    :type split_dfs: dict
    :param crypto_name: The name of the cryptocurrency whose output will be used to train a dl NN
    :type crypto_name: str
    :return: a tuple of (training_inputs, training_outputs, test_inputs, test_outputs)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    training_inputs = []
    training_outputs = np.array(split_dfs[crypto_name]['y_dev'])
    test_inputs = []
    test_outputs = np.array(split_dfs[crypto_name]['y_eval'])
    for (crypto, dfs) in split_dfs.iteritems():
        training_inputs.append(dfs['X_dev'])
        test_inputs.append(dfs['X_eval'])

    training_inputs = [np.array(training_input) for training_input in training_inputs]
    training_inputs = np.array(training_inputs)
    training_inputs = np.swapaxes(training_inputs, 0, 1)
    training_inputs = np.swapaxes(training_inputs, 1, 2)

    test_inputs = [np.array(test_inputs) for test_inputs in test_inputs]
    test_inputs = np.array(test_inputs)
    test_inputs = np.swapaxes(test_inputs, 0, 1)
    test_inputs = np.swapaxes(test_inputs, 1, 2)

    return training_inputs, training_outputs, test_inputs, test_outputs


def build_model(input_shape, output_size, neurons, activation_func="sigmoid",
                dropout=0.25, loss="mae", optimizer="adam"):
    """
    Compile a Sequential Keras deep learning model given a set of parameters.
    The model has an LSTM layer.
    :param input_shape: The input layer shape (not counting sample size)
    :type input_shape: tuple
    :param output_size: The number of units in the output layer (usually 1)
    :type output_size: int
    :param neurons: The number of neurons in the LSTM layer
    :type neurons: int
    :param activation_func: The activation function name (such as 'sigmoid').
    :type activation_func: str
    :param dropout: The dropout value to reduce overfitting and improve generalization
    :type dropout: float
    :param loss: loss function (such as 'mae')
    :type loss: str
    :param optimizer: optimizer (such as 'adam')
    :type optimizer: str
    :return: a compiled sequential model
    :rtype: Sequential
    """
    model = Sequential()

    model.add(LSTM(neurons, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activation_func))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def run_deep_learning_pipeline(crypto_name, split_dfs, deep_learning_config):
    """
    Run the deep learning pipeline with the given split DataFrames per cryptocurrency. Each cryptocurrency's DataFrame
    is used as an input channel for the NN.
    The function returns an optimized and trained deep neural network model.
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.)
    :type crypto_name: str
    :param split_dfs: A dictionary of split DataFrames per cryptocurrency.
    :type split_dfs: dict
    :param deep_learning_config
    :type deep_learning_config: DeepLearningConfig
    :return: a dictionary with the trained model and performance metrics {'model': model, 'metrics': {}}
    :rtype: dict
    """

    logger.info("Running deep learning pipeline...")
    logger.info("Deep learning config:\n" + str(deep_learning_config))

    # Build datasets
    training_inputs, training_outputs, test_inputs, test_outputs = \
        build_deep_learning_datasets(split_dfs, crypto_name)

    # Initialize model architecture
    logger.info("Initializing model architecture...")
    model = build_model(input_shape=(training_inputs.shape[1], training_inputs.shape[2]), output_size=1,
                        neurons=deep_learning_config.neurons)

    # Train model on data
    logger.info("Training model on data...")
    model.fit(training_inputs, training_outputs, epochs=100, batch_size=1, verbose=2, shuffle=True)

    # Evaluate model
    metric_values = model.evaluate(test_inputs, test_outputs, verbose=2)
    metrics = {metric_name: metric_value for (metric_name, metric_value) in zip(model.metrics_names, metric_values)}
    logger.info("Metrics:\n{}".format(metrics))

    logger.info("Deep learning pipeline complete!\n")

    model_dict = {
        'model': model,
        'metrics': metrics
    }

    return model_dict


def _get_dl_run_name(crypto, preprocessing_config, training_config, deep_learning_config):
    """
    Get a string with the name of the deep-learning run given preprocessing, training and deep learning configurations.

    >>> _get_dl_run_name('LTC',
    ...     PreprocessingConfig(
    ...     {
    ...         'window_size': 60,
    ...         'predictor_params': {
    ...             'lookahead_days': 3, 'prob_buy': 1, 'prob_sell': 1, 'starting_investment': 100, 'daily_allowance': 5
    ...         },
    ...         'start_date': '2018-01-01',
    ...         'end_date': '2019-05-04',
    ...         'price_column': 'Close',
    ...         'normalize': True,
    ...         'normalize_by_row': True,
    ...         'standardize': False,
    ...         'predictor_cls': 'BiffPredictor',
    ...     }),
    ...     TrainingConfig(
    ...     {
    ...         'dev_size': 0.5,
    ...         'shuffle_data': True,
    ...         'training_size': 0.7,
    ...         'cv_folds': 4
    ...     }),
    ...     DeepLearningConfig(
    ...     {
    ...         'neurons': 10
    ...     }
    ...     )
    ... )
    'LTC_2019-05-04TrueTrueBiffPredictor5311100CloseFalse2018-01-0160_40.5True0.7_10'

    :param crypto: The crypto name
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :param deep_learning_config
    :type deep_learning_config: DeepLearningConfig
    :return: A file name
    :rtype: str
    """

    return "{}_{}_{}_{}".format(crypto, preprocessing_config.to_single_line_str(), training_config.to_single_line_str(),
                                deep_learning_config.to_single_line_str())


def get_model_name(crypto, preprocessing_config, training_config, deep_learning_config):
    """
    Get a string with the name of a trained deep-learning model given preprocessing, training and deep learning
    configurations.
    The saved model contains architecture, weights, and optimizer state.

    >>> get_model_name('LTC',
    ...     PreprocessingConfig(
    ...     {
    ...         'window_size': 60,
    ...         'predictor_params': {
    ...             'lookahead_days': 3, 'prob_buy': 1, 'prob_sell': 1, 'starting_investment': 100, 'daily_allowance': 5
    ...         },
    ...         'start_date': '2018-01-01',
    ...         'end_date': '2019-05-04',
    ...         'price_column': 'Close',
    ...         'normalize': True,
    ...         'normalize_by_row': True,
    ...         'standardize': False,
    ...         'predictor_cls': 'BiffPredictor',
    ...     }),
    ...     TrainingConfig(
    ...     {
    ...         'dev_size': 0.5,
    ...         'shuffle_data': True,
    ...         'training_size': 0.7,
    ...         'cv_folds': 4
    ...     }),
    ...     DeepLearningConfig(
    ...     {
    ...         'neurons': 10
    ...     }
    ...     )
    ... )
    'LSTM_LTC_2019-05-04TrueTrueBiffPredictor5311100CloseFalse2018-01-0160_40.5True0.7_10.h5'

    :param crypto: The crypto name
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :param deep_learning_config
    :type deep_learning_config: DeepLearningConfig
    :return: A file name
    :rtype: str
    """
    file_type = 'h5'
    return "LSTM_{}_{}_{}_{}.{}".format(crypto, preprocessing_config.to_single_line_str(),
                                        training_config.to_single_line_str(), deep_learning_config.to_single_line_str(),
                                        file_type)


def save_deep_learning_results(model_dict, crypto, preprocessing_config, training_config, deep_learning_config):
    """
    Save training results to a file.
    :param model_dict: dictionary containing actual trained model and a dictionary of its evaluation metrics
    :type model_dict: dict
    :param crypto: The name of the crypto
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :param deep_learning_config
    :type deep_learning_config: DeepLearningConfig
    """

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    # Human-readable
    hr_file_name = "results_dl_{}.txt".format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    logger.info("Saving results to {}...".format(os.path.join(RESULTS_DIR, hr_file_name)))
    with open(os.path.join(RESULTS_DIR, hr_file_name), 'a') as f:
        f.write(crypto + '\n')
        f.write(str(preprocessing_config) + '\n')
        f.write(str(training_config) + '\n')
        f.write(str(deep_learning_config) + '\n')
        f.write(str(model_dict) + '\n')
        f.write('\n**************************************************************\n')

    # CSV
    csv_file_name = "results_dl_{}.csv".format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    logger.info("Saving results to {}...".format(os.path.join(RESULTS_DIR, csv_file_name)))
    metrics = sorted(model_dict['metrics'].keys())
    metrics_header = ','.join(["LSTM_{}".format(m) for m in metrics])
    metrics_values = [str(model_dict['metrics'][m]) for m in metrics]
    metrics_csv = ','.join(metrics_values)
    headers = "{},{},{},{},{}".format('crypto', preprocessing_config.to_csv_str()[0], training_config.to_csv_str()[0],
                                      deep_learning_config.to_csv_str()[0], metrics_header)
    line = "{},{},{},{},{}".format(crypto, preprocessing_config.to_csv_str()[1], training_config.to_csv_str()[1],
                                   deep_learning_config.to_csv_str()[1], metrics_csv)

    file_exists = os.path.isfile(os.path.join(RESULTS_DIR, csv_file_name))
    with open(os.path.join(RESULTS_DIR, csv_file_name), 'a') as f:
        if not file_exists:
            f.write(headers + '\n')

        f.write(line + '\n')

    training_run_touch_file = _get_dl_run_name(crypto, preprocessing_config, training_config, deep_learning_config)
    with open(os.path.join(RESULTS_DIR, training_run_touch_file), 'w') as f:
        f.write('complete')


def save_deep_learning_model(model, crypto, preprocessing_config, training_config, deep_learning_config,
                             model_dir=None):
    """
    Save a deep learning model (NN) and its weights to disk.
    :param model: The trained model
    :type model: Sequential
    :param crypto
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :param deep_learning_config
    :type deep_learning_config: DeepLearningConfig
    :param model_dir: (Default None) The (overridden) directory where the model should be saved.
    If None, the default `MODELS_DIR` is used.
    :type model_dir: str
    """
    dir_to_use = MODELS_DIR if model_dir is None else model_dir
    if not os.path.exists(dir_to_use):
        os.mkdir(dir_to_use)

    model_file = get_model_name(crypto, preprocessing_config, training_config, deep_learning_config)
    model_file = os.path.join(dir_to_use, model_file)
    model.save(model_file)

    logger.info("Saved model to disk!\n")


def load_deep_learning_model(crypto, preprocessing_config, training_config, deep_learning_config,
                             model_dir=None):
    """
    Load a saved and trained deep learning model (NN) from disk.
    :param crypto
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :param deep_learning_config
    :type deep_learning_config: DeepLearningConfig
    :param model_dir: (Default None) The (overridden) directory where the model should be loaded from.
    If None, the default `MODELS_DIR` is used.
    :type model_dir: str
    :return: A fitted model loaded from disk
    :rtype: Sequential
    """
    dir_to_use = MODELS_DIR if model_dir is None else model_dir
    if not os.path.exists(dir_to_use):
        os.mkdir(dir_to_use)

    model_file = get_model_name(crypto, preprocessing_config, training_config, deep_learning_config)
    model_file = os.path.join(dir_to_use, model_file)
    model = load_model(model_file)

    return model


def run_deep_learning(cryptonalysis_config_grid):
    """
    Run deep learning training using grid search hyperparameter optimization.
    :param cryptonalysis_config_grid
    :type cryptonalysis_config_grid: CryptonalysisConfigGrid
    """

    logger.info("Config grid size: {}".format(cryptonalysis_config_grid.grid_size))

    # Grid search preprocessing pipeline parameters
    count = 1
    for crypto in cryptonalysis_config_grid.cryptos:
        logger.info("Crypto: {}".format(crypto))
        for preprocessing_config in cryptonalysis_config_grid.preprocessing_config_grid:
            preprocessed_dfs = {
                crypto_name: run_preprocessing_pipeline(crypto_name, preprocessing_config,
                                                        cryptonalysis_config_grid.save_preprocessing_data,
                                                        cryptonalysis_config_grid.save_preprocessing_roi)
                for crypto_name in CRYPTOCURRENCIES
            }

            # Grid search training pipeline parameters
            for training_config in cryptonalysis_config_grid.training_config_grid:
                split_dfs = get_split_dfs(preprocessed_dfs, training_config)

                for deep_learning_config in cryptonalysis_config_grid.deep_learning_config_grid:
                    logger.info("Processing configuration {}/{}...".format(count, cryptonalysis_config_grid.grid_size))
                    count += 1

                    # Check if training run has been executed
                    if cryptonalysis_config_grid.save_training_results:
                        training_run_touch_file = _get_dl_run_name(crypto, preprocessing_config, training_config,
                                                                   deep_learning_config)
                        if os.path.exists(os.path.join(RESULTS_DIR, training_run_touch_file)):
                            logger.info("Training run has already been executed, skipping...")
                            continue

                    model_dict = run_deep_learning_pipeline(crypto, split_dfs, deep_learning_config)
                    if cryptonalysis_config_grid.save_training_results:
                        save_deep_learning_results(model_dict, crypto, preprocessing_config, training_config,
                                                   deep_learning_config)
                    if cryptonalysis_config_grid.save_training_model:
                        model = model_dict['model']
                        save_deep_learning_model(model, crypto, preprocessing_config, training_config,
                                                 deep_learning_config)


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Config file not given in args!')

    config_file = sys.argv[1]

    config_path = 'config'
    config_grid = load_cryptonalysis_config_grid(config_path, config_file)

    run_deep_learning(config_grid)
