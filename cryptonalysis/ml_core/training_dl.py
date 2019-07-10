# Methods for deep learning

import logging
import numpy as np
import os
from config import load_cryptonalysis_config_grid, TrainingConfig, CryptonalysisConfigGrid
from preprocessing import run_preprocessing_pipeline, CRYPTOCURRENCIES
from training import split_datasets
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Logging
logger = logging.getLogger()


def build_model(inputs, output_size, neurons, activation_func="sigmoid",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activation_func))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


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


def run_deep_learning_pipeline(crypto_name, preprocessed_dfs, training_config, dev_size=0.5):
    """
    Run the deep learning pipeline. The function returns an optimized and trained classification model.
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.)
    :type crypto_name: str
    :param preprocessed_dfs: The preprocessed DataFrames ready for the classification pipeline
    :param training_config
    :type training_config: TrainingConfig
    :param dev_size: The size (0~1) of the development dataset (used in Grid Search CV)
    :return:
    """

    logger.info("Running deep learning pipeline...")
    logger.info("Training config:\n" + str(training_config))

    # 1. Split datasets for classification
    logger.info("Splitting datasets for classification...")
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
        _, _, _, _, _, _, X_dev, y_dev, X_eval, y_eval = split_datasets(df, shuffle=training_config.shuffle_data,
                                                                        training_size=0.7, dev_size=0.5)
        split_dfs[crypto]['X_dev'] = X_dev
        split_dfs[crypto]['y_dev'] = y_dev
        split_dfs[crypto]['X_eval'] = X_eval
        split_dfs[crypto]['y_eval'] = y_eval

    training_inputs, training_outputs, test_inputs, test_outputs = \
        build_deep_learning_datasets(split_dfs, crypto_name)

    # 2. Initialize model architecture
    logger.info("Initializing model architecture...")
    model = build_model(training_inputs, output_size=1, neurons=30)

    # 3. Train model on data
    logger.info("Training model on data...")
    model.fit(training_inputs, training_outputs, epochs=100, batch_size=1, verbose=2, shuffle=True)

    # 4. Evaluate model
    scores = model.evaluate(test_inputs, test_outputs, verbose=2)
    logger.info("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # 5. Serialize model to JSON
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

    # Grid search preprocessing pipeline parameters
    for crypto in cryptonalysis_config_grid.cryptos:
        for preprocessing_config in cryptonalysis_config_grid.preprocessing_config_grid:
            preprocessed_dfs = {
                crypto_name: run_preprocessing_pipeline(crypto_name, preprocessing_config)
                for crypto_name in CRYPTOCURRENCIES
            }

            # Grid search training pipeline parameters
            for training_config in cryptonalysis_config_grid.training_config_grid:
                run_deep_learning_pipeline(crypto, preprocessed_dfs, training_config)


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    config_path = os.path.join(os.path.pardir, os.path.join(os.path.pardir, 'config'))
    config_grid = load_cryptonalysis_config_grid(config_path)

    run_deep_learning(config_grid)
