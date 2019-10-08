# Methods for deep learning

import logging
import numpy as np
import os
import sys
from config import load_cryptonalysis_config_grid, TrainingConfig, DeepLearningConfig, CryptonalysisConfigGrid
from preprocessing import run_preprocessing_pipeline, CRYPTOCURRENCIES
from training import split_datasets
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Logging
logger = logging.getLogger()


def build_deep_learning_datasets(split_dfs, crypto_name):
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


def build_model(inputs, output_size, neurons, activation_func="sigmoid",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activation_func))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


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
    for (crypto, df) in preprocessed_dfs.iteritems():
        logger.info("*** {} ***".format(crypto))
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(df, False,  # TODO: Should shuffle data! Fix idx
                                                                        training_config.training_size,
                                                                        training_config.dev_size)
        split_dfs[crypto]['X_dev'] = X_dev
        split_dfs[crypto]['y_dev'] = y_dev
        split_dfs[crypto]['X_eval'] = X_eval
        split_dfs[crypto]['y_eval'] = y_eval

    return split_dfs


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
    :return: a trained model for the given configuration
    """

    logger.info("Running deep learning pipeline...")
    logger.info("Deep learning config:\n" + str(deep_learning_config))

    # Build datasets
    training_inputs, training_outputs, test_inputs, test_outputs = \
        build_deep_learning_datasets(split_dfs, crypto_name)

    # Initialize model architecture
    logger.info("Initializing model architecture...")
    model = build_model(training_inputs, output_size=1, neurons=deep_learning_config.neurons)

    # Train model on data
    logger.info("Training model on data...")
    model.fit(training_inputs, training_outputs, epochs=100, batch_size=1, verbose=2, shuffle=True)

    # Evaluate model
    scores = model.evaluate(test_inputs, test_outputs, verbose=2)
    logger.info("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # Serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("model.h5")
    logger.info("Saved model to disk!\n")

    logger.info("Deep learning pipeline complete!\n")

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
                    run_deep_learning_pipeline(crypto, split_dfs, deep_learning_config)
                    count += 1


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Config file not given in args!')

    config_file = sys.argv[1]

    config_path = os.path.join(os.path.pardir, os.path.join(os.path.pardir, 'config'))
    config_grid = load_cryptonalysis_config_grid(config_path, config_file)

    run_deep_learning(config_grid)
