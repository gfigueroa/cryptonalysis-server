# Methods for deep learning

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


def build_model(inputs, output_size, neurons, activation_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activation_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


def build_deep_learning_datasets(split_dfs, crypto_name):
    lstm_training_inputs = []
    lstm_training_outputs = np.array(split_dfs[crypto_name]['y_dev'])
    lstm_test_inputs = []
    lstm_test_outputs = np.array(split_dfs[crypto_name]['y_eval'])
    for (crypto, dfs) in split_dfs.iteritems():
        lstm_training_inputs.append(dfs['X_dev'])
        lstm_test_inputs.append(dfs['X_eval'])

    lstm_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in lstm_training_inputs]
    lstm_training_inputs = np.array(lstm_training_inputs)
    lstm_training_inputs = np.swapaxes(lstm_training_inputs, 0, 1)
    lstm_training_inputs = np.swapaxes(lstm_training_inputs, 1, 2)

    lstm_test_inputs = [np.array(lstm_test_inputs) for lstm_test_inputs in lstm_test_inputs]
    lstm_test_inputs = np.array(lstm_test_inputs)
    lstm_test_inputs = np.swapaxes(lstm_test_inputs, 0, 1)
    lstm_test_inputs = np.swapaxes(lstm_test_inputs, 1, 2)

    return lstm_training_inputs, lstm_training_outputs, lstm_test_inputs, lstm_test_outputs
