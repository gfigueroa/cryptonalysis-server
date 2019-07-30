# -*- coding: utf-8 -*-
"""
Configuration classes.
"""

import json
import os
from datetime import date, datetime
from logging.config import fileConfig
from pyhocon import ConfigFactory, ConfigTree
from cryptonalysis.utils import misc_utils
from cryptonalysis.ml_core.transaction_builders import get_predictor_class_from_name
from pandas.io.json import json_normalize


class Config(object):

    def __init__(self, config_dict):
        self.config_dict = config_dict

    def to_csv_str(self):
        flat_config_dict = json_normalize(self.config_dict).to_dict(orient='records')[0]
        sorted_keys = sorted(flat_config_dict.keys())
        values = ','.join([str(flat_config_dict[k]) for k in sorted_keys])
        return sorted_keys, values

    def to_single_line_str(self):
        flat_config_dict = json_normalize(self.config_dict).to_dict(orient='records')[0]
        sorted_keys = sorted(flat_config_dict.keys())
        values = ''.join([str(flat_config_dict[k]) for k in sorted_keys])
        return values

    def __str__(self):
        return json.dumps(self.config_dict, indent=2)


class PreprocessingConfig(Config):

    # Default values
    START_DATE = misc_utils.parse_date('2015-08-07')  # The date from which to start historical data preprocessing
    END_DATE = misc_utils.parse_date('today')  # The date in which to end historical data preprocessing
    WINDOW_SIZE = 10  # The window size in days used by the predictor (for sliding window training technique)
    NORMALIZE = True  # Whether or not to normalize the historical data
    NORMALIZE_BY_ROW = False  # Normalize by row using feature scaling or use scikit-learn's FeatureScaler
    PRICE_COLUMN = 'Close'  # The column in the historical dataset used for training and testing
    SAVE_ROI = False  # Whether or not to save the ROI to a local file for analysis
    SAVE_DATA = False  # Whether or not to save preprocessed data to a local file to avoid recalculation
    PREDICTOR_CLS = get_predictor_class_from_name('BiffPredictor')  # The class used to build transactions
    PREDICTOR_PARAMS = {
        'prob_buy': 1,  # A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
        'prob_sell': 1,  # A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
        'starting_investment': 100,  # How much money to start with (used for ROI simulations)
        'daily_allowance': 5,  # How much money to invest every day (used for ROI simulations)
        'lookahead_days': 4  # The number of days to look ahead when making a transaction
    }

    def __init__(self, preprocessing_config_dict):
        """
        Initialize PreprocessingConfig object with a dictionary of values.
        :param preprocessing_config_dict
        :type preprocessing_config_dict: dict
        """

        super(PreprocessingConfig, self).__init__(preprocessing_config_dict)

        if 'start_date' in preprocessing_config_dict:
            start_date = preprocessing_config_dict['start_date']
            self.start_date = start_date if type(start_date) is date else misc_utils.parse_date(start_date)
        else:
            self.start_date = PreprocessingConfig.START_DATE
            preprocessing_config_dict['start_date'] = self.start_date.strftime('%Y-%m-%d')

        if 'end_date' in preprocessing_config_dict:
            end_date = preprocessing_config_dict['end_date']
            self.end_date = end_date if type(end_date) is date else misc_utils.parse_date(end_date)
        else:
            self.end_date = PreprocessingConfig.END_DATE
            preprocessing_config_dict['end_date'] = self.end_date.strftime('%Y-%m-%d')

        self.window_size = preprocessing_config_dict['window_size'] = preprocessing_config_dict['window_size'] \
            if 'window_size' in preprocessing_config_dict else PreprocessingConfig.WINDOW_SIZE

        self.normalize = preprocessing_config_dict['normalize'] = preprocessing_config_dict['normalize'] \
            if 'normalize' in preprocessing_config_dict else PreprocessingConfig.NORMALIZE

        self.normalize_by_row = preprocessing_config_dict['normalize_by_row'] = \
            preprocessing_config_dict['normalize_by_row'] \
            if 'normalize_by_row' in preprocessing_config_dict else PreprocessingConfig.NORMALIZE_BY_ROW

        self.price_column = preprocessing_config_dict['price_column'] = preprocessing_config_dict['price_column'] \
            if 'price_column' in preprocessing_config_dict else PreprocessingConfig.PRICE_COLUMN

        self.save_roi = preprocessing_config_dict['save_roi'] = preprocessing_config_dict['save_roi'] \
            if 'save_roi' in preprocessing_config_dict else PreprocessingConfig.SAVE_ROI

        self.save_data = preprocessing_config_dict['save_data'] = preprocessing_config_dict['save_data'] \
            if 'save_data' in preprocessing_config_dict else PreprocessingConfig.SAVE_DATA

        if 'predictor_cls' in preprocessing_config_dict:
            predictor_cls = preprocessing_config_dict['predictor_cls']
            self.predictor_cls = \
                predictor_cls if type(predictor_cls) is type else get_predictor_class_from_name(predictor_cls)
        else:
            self.predictor_cls = PreprocessingConfig.PREDICTOR_CLS
            preprocessing_config_dict['predictor_cls'] = self.predictor_cls.__name__

        self.predictor_params = preprocessing_config_dict['predictor_params'] = \
            preprocessing_config_dict['predictor_params'] \
            if 'predictor_params' in preprocessing_config_dict else PreprocessingConfig.PREDICTOR_PARAMS

        self.config_dict = preprocessing_config_dict

    def adjust_end_date(self, latest_date):
        """
        Adjust the end_date if the latest date is earlier.
        :param latest_date
        :type latest_date: datetime
        """
        if latest_date < self.end_date:
            self.end_date = latest_date
            self.config_dict['end_date'] = datetime.strftime(latest_date, '%Y-%m-%d')

    def __str__(self):
        return json.dumps(self.config_dict, indent=2)


class TrainingConfig(Config):

    # Default values
    SHUFFLE_DATA = True  # Whether or not to shuffle the rows of the training data
    TRAINING_SIZE = 0.7  # The size (0~1) of the training dataset (used in non-CV). Remaining is for testing.
    DEV_SIZE = 0.5  # The size (0~1) of the development dataset (used in Grid Search CV). Remaining is for evaluation.
    CV_FOLDS = 4  # Number of folds (k) used in cross validation
    SAVE_RESULTS = False  # Whether or not to save results to a local file
    SAVE_MODEL = False  # Whether or not to save model to a local file

    def __init__(self, training_config_dict):
        """
        Initialize TrainingConfig object with a dictionary of values.
        :param training_config_dict
        :type training_config_dict: dict
        """

        super(TrainingConfig, self).__init__(training_config_dict)

        self.shuffle_data = training_config_dict['shuffle_data'] = training_config_dict['shuffle_data'] \
            if 'shuffle_data' in training_config_dict else TrainingConfig.SHUFFLE_DATA
        self.training_size = training_config_dict['training_size'] = float(training_config_dict['training_size']) \
            if 'training_size' in training_config_dict else TrainingConfig.TRAINING_SIZE
        self.dev_size = training_config_dict['dev_size'] = float(training_config_dict['dev_size']) \
            if 'dev_size' in training_config_dict else TrainingConfig.DEV_SIZE
        self.cv_folds = training_config_dict['cv_folds'] = int(training_config_dict['cv_folds']) \
            if 'cv_folds' in training_config_dict else TrainingConfig.CV_FOLDS
        self.save_results = training_config_dict['save_results'] = training_config_dict['save_results'] \
            if 'save_results' in training_config_dict else TrainingConfig.SAVE_RESULTS
        self.save_model = training_config_dict['save_model'] = training_config_dict['save_model'] \
            if 'save_model' in training_config_dict else TrainingConfig.SAVE_MODEL

        self.config_dict = training_config_dict


class DeepLearningConfig(Config):

    # Default values
    NEURONS = 30  # Number of neurons used to build NN model

    def __init__(self, deep_learning_config_dict):
        """
        Initialize DeepLearningConfig object with a dictionary of values.
        :param deep_learning_config_dict
        :type deep_learning_config_dict: dict
        """
        super(DeepLearningConfig, self).__init__(deep_learning_config_dict)

        self.neurons = deep_learning_config_dict['neurons'] = deep_learning_config_dict['neurons'] \
            if 'neurons' in deep_learning_config_dict else DeepLearningConfig.NEURONS

        self.config_dict = deep_learning_config_dict


class CryptonalysisConfig(Config):

    # Default values
    CRYPTO = 'ETH'
    PREPROCESSING_CONFIG = PreprocessingConfig({})
    TRAINING_CONFIG = TrainingConfig({})
    DEEP_LEARNING_CONFIG = DeepLearningConfig({})

    def __init__(self, cryptonalysis_config_dict):
        """
        Initialize CryptonalysisConfig object.
        :param cryptonalysis_config_dict
        :type cryptonalysis_config_dict: dict
        """
        super(CryptonalysisConfig, self).__init__(cryptonalysis_config_dict)

        self.crypto = cryptonalysis_config_dict['crypto'] = cryptonalysis_config_dict['crypto'] \
            if 'crypto' in cryptonalysis_config_dict else CryptonalysisConfig.CRYPTO

        self.preprocessing = PreprocessingConfig(cryptonalysis_config_dict['preprocessing']) \
            if 'preprocessing' in cryptonalysis_config_dict else CryptonalysisConfig.PREPROCESSING_CONFIG
        cryptonalysis_config_dict['preprocessing'] = self.preprocessing.config_dict

        self.training = TrainingConfig(cryptonalysis_config_dict['training']) \
            if 'training' in cryptonalysis_config_dict else CryptonalysisConfig.TRAINING_CONFIG
        cryptonalysis_config_dict['training'] = self.training.config_dict

        self.deep_learning = DeepLearningConfig(cryptonalysis_config_dict['deep_learning']) \
            if 'deep_learning' in cryptonalysis_config_dict else CryptonalysisConfig.DEEP_LEARNING_CONFIG
        cryptonalysis_config_dict['deep_learning'] = self.deep_learning.config_dict

        self.config_dict = cryptonalysis_config_dict


class CryptonalysisConfigGrid(object):

    def __init__(self, cryptos, preprocessing_config_grid, training_config_grid, deep_learning_config_grid):
        """
        Initialize CryptonalysisConfigGrid object.
        :param cryptos: List of cryptocurrencies to use (e..g ETH, BTC, etc.)
        :type cryptos: list of str
        :param preprocessing_config_grid
        :type preprocessing_config_grid: list of PreprocessingConfig
        :param training_config_grid
        :type training_config_grid: list of TrainingConfig
        :param deep_learning_config_grid: list of DeepLearningConfig
        :type deep_learning_config_grid: list of DeepLearningConfig
        """
        if not cryptos:
            raise ValueError("cryptos list is required in CryptonalysisConfigGrid!")

        self.cryptos = cryptos
        self.preprocessing_config_grid = preprocessing_config_grid
        self.training_config_grid = training_config_grid
        self.deep_learning_config_grid = deep_learning_config_grid
        self.grid_size = len(cryptos) * (len(preprocessing_config_grid) or 1) * (len(training_config_grid) or 1) * \
                         (len(deep_learning_config_grid) or 1)


def convert_to_flat_dict(config, prefix=""):
    """
    Recursively convert ConfigTree or dictionary into a flat dictionary.
    :param config
    :type config: ConfigTree or dict
    :param prefix: Prefix to add to each key when flattening
    :return: A flat configuration tree
    :rtype: dict
    """
    result = {}
    for k in config:
        v = config[k]
        if isinstance(v, dict):
            for k1 in convert_to_flat_dict(v, k + "."):
                result[prefix + k1] = str(config[k1])
        else:
            result[prefix + k] = str(v)
    return result


def convert_to_dict(config):
    """
    Recursively convert ConfigTree into a dictionary.
    :param config
    :type config: ConfigTree or dict
    :return: A dictionary
    :rtype: dict
    """
    result = {}
    for k in config:
        v = config[k]
        if isinstance(v, dict) or isinstance(v, ConfigTree):
            v = convert_to_dict(v)
        result[k] = v
    return result


def build_cryptonalysis_config_grid(config):
    """
    Build a CryptonalysisConfigGrid object from a config dictionary.
    The CryptonalysisConfigGrid object is composed of a list of PreprocessingConfig objects and a list of TrainingConfig
    objects.
    :param config
    :type config: dict
    :return: An instance of CryptonalysisConfigGrid.
    :rtype: CryptonalysisConfigGrid
    """

    def _build_config_grid(conf, grid=None):
        temp_config = conf.copy()
        if grid is None:
            grid = []

            # Build dictionary lists
            for param, value in conf.items():
                if type(value) is dict:
                    temp_config[param] = _build_config_grid(value)

        for param, value in temp_config.items():
            if type(value) is list:
                for sub_value in value:
                    temp_config[param] = sub_value
                    _build_config_grid(temp_config, grid)
            else:
                temp_config[param] = value

        if temp_config not in grid:
            grid.append(temp_config)

        return grid

    cryptos = config['crypto'] if type(config['crypto']) is list else [config['crypto']]
    preprocessing_config_grid = [PreprocessingConfig(pc) for pc in _build_config_grid(config['preprocessing'])]
    training_config_grid = [TrainingConfig(tc) for tc in _build_config_grid(config['training'])] \
        if 'training' in config else []
    deep_learning_config_grid = [DeepLearningConfig(dlc) for dlc in _build_config_grid(config['deep_learning'])] \
        if 'deep_learning' in config else []

    cryptonalysis_config_grid = CryptonalysisConfigGrid(
        cryptos=cryptos,
        preprocessing_config_grid=preprocessing_config_grid,
        training_config_grid=training_config_grid,
        deep_learning_config_grid=deep_learning_config_grid
    )

    return cryptonalysis_config_grid


def load_config(config_path, config_file_name):
    config_file = os.path.join(config_path, config_file_name)

    logging_config_file_name = 'logging.ini'
    logging_config_file = os.path.join(config_path, logging_config_file_name)

    config = ConfigFactory.parse_file(config_file)

    if logging_config_file:
        flat_config = convert_to_flat_dict(config)
        fileConfig(fname=logging_config_file, defaults=flat_config)

    return config


def load_cryptonalysis_config(config_path, config_file_name):
    config = load_config(config_path, config_file_name)

    config_dict = convert_to_dict(config)
    cryptonalysis_config = CryptonalysisConfig(config_dict)

    return cryptonalysis_config


def load_cryptonalysis_config_grid(config_path, config_file_name):
    config = load_config(config_path, config_file_name)

    config_dict = convert_to_dict(config)
    cryptonalysis_config_grid = build_cryptonalysis_config_grid(config_dict)

    return cryptonalysis_config_grid


if __name__ == '__main__':
    pcg = load_cryptonalysis_config_grid('config', 'preprocessing.conf')
    for pc in pcg.preprocessing_config_grid:
        print pc
        print pc.to_csv_str()
    for tc in pcg.training_config_grid:
        print tc
        print tc.to_csv_str()
    for dlc in pcg.deep_learning_config_grid:
        print dlc
        print dlc.to_csv_str()

    tcg = load_cryptonalysis_config_grid('config', 'training.conf')
    print tcg

    dlc = load_cryptonalysis_config_grid('config', 'training_dl.conf')
    print dlc
