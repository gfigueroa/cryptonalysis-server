# -*- coding: utf-8 -*-
"""
Configuration classes.
"""

import os
from datetime import date
from logging.config import fileConfig
from pyhocon import ConfigFactory, ConfigTree
from cryptonalysis.utils import misc_utils
from cryptonalysis.ml_core.transaction_builders import get_predictor_class_from_name


class PreprocessingConfig(object):

    # Default values
    START_DATE = misc_utils.parse_date('2015-08-07')
    END_DATE = misc_utils.parse_date('today')
    WINDOW_SIZE = 10
    NORMALIZE = True
    NORMALIZE_BY_ROW = False
    PRICE_COLUMN = 'Close'
    SAVE_ROI = False
    SAVE_DATA = False
    PREDICTOR_CLS = get_predictor_class_from_name('BiffPredictor')
    PREDICTOR_PARAMS = {
        'prob_buy': 1,
        'prob_sell': 1,
        'starting_investment': 100,
        'daily_allowance': 5,
        'lookahead_days': 4
    }

    def __init__(self, preprocessing_config_dict):
        """
        Initialize PreprocessingConfig object with a dictionary of values.
        :param preprocessing_config_dict
        :type preprocessing_config_dict: dict
        """
        if 'start_date' in preprocessing_config_dict:
            start_date = preprocessing_config_dict['start_date']
            self.start_date = start_date if type(start_date) is date else misc_utils.parse_date(start_date)
        else:
            self.start_date = PreprocessingConfig.START_DATE

        if 'end_date' in preprocessing_config_dict:
            end_date = preprocessing_config_dict['end_date']
            self.end_date = end_date if type(end_date) is date else misc_utils.parse_date(end_date)
        else:
            self.end_date = PreprocessingConfig.END_DATE

        self.window_size = preprocessing_config_dict['window_size'] \
            if 'window_size' in preprocessing_config_dict else PreprocessingConfig.WINDOW_SIZE

        self.normalize = preprocessing_config_dict['normalize'] \
            if 'normalize' in preprocessing_config_dict else PreprocessingConfig.NORMALIZE

        self.normalize_by_row = preprocessing_config_dict['normalize_by_row'] \
            if 'normalize_by_row' in preprocessing_config_dict else PreprocessingConfig.NORMALIZE_BY_ROW

        self.price_column = preprocessing_config_dict['price_column'] \
            if 'price_column' in preprocessing_config_dict else PreprocessingConfig.PRICE_COLUMN

        self.save_roi = preprocessing_config_dict['save_roi'] \
            if 'save_roi' in preprocessing_config_dict else PreprocessingConfig.SAVE_ROI

        self.save_data = preprocessing_config_dict['save_data'] \
            if 'save_data' in preprocessing_config_dict else PreprocessingConfig.SAVE_DATA

        if 'predictor_cls' in preprocessing_config_dict:
            predictor_cls = preprocessing_config_dict['predictor_cls']
            self.predictor_cls = \
                predictor_cls if type(predictor_cls) is type else get_predictor_class_from_name(predictor_cls)
        else:
            self.predictor_cls = PreprocessingConfig.PREDICTOR_CLS

        self.predictor_params = preprocessing_config_dict['predictor_params'] \
            if 'predictor_params' in preprocessing_config_dict else PreprocessingConfig.PREDICTOR_PARAMS


class TrainingConfig(object):

    # Default values
    SHUFFLE_DATA = True

    def __init__(self, training_config_dict):
        """
        Initialize TrainingConfig object with a dictionary of values.
        :param training_config_dict
        :type training_config_dict: dict
        """
        self.shuffle_data = training_config_dict['shuffle_data'] \
            if 'shuffle_data' in training_config_dict else TrainingConfig.SHUFFLE_DATA


class CryptonalysisConfig(object):

    def __init__(self, crypto, preprocessing_config=None, training_config=None):
        """
        Initialize ClassificationConfig object.
        :param crypto: The cryptocurrency to use (e..g ETH, BTC, etc.)
        :type crypto: str
        :param preprocessing_config
        :type preprocessing_config: PreprocessingConfig
        :param training_config
        :type training_config: TrainingConfig
        """
        self.crypto = crypto
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config


class CryptonalysisConfigGrid(object):

    def __init__(self, crypto, preprocessing_config_grid, training_config_grid):
        """
        Initialize CryptonalysisConfigGrid object.
        :param crypto: The cryptocurrency to use (e..g ETH, BTC, etc.)
        :type crypto: str
        :param preprocessing_config_grid
        :type preprocessing_config_grid: list of PreprocessingConfig
        :param training_config_grid
        :type training_config_grid: list of TrainingConfig
        """
        self.crypto = crypto
        self.preprocessing_config = preprocessing_config_grid
        self.training_config = training_config_grid


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

    preprocessing_config_grid = [PreprocessingConfig(pc) for pc in _build_config_grid(config['preprocessing'])]
    training_config_grid = [TrainingConfig(tc) for tc in _build_config_grid(config['training'])]

    cryptonalysis_config_grid = CryptonalysisConfigGrid(
        crypto=config['crypto'],
        preprocessing_config_grid=preprocessing_config_grid,
        training_config_grid=training_config_grid
    )

    return cryptonalysis_config_grid


def load_preprocessing_config(config_path):
    config_file_name = 'preprocessing.conf'
    config_file = os.path.join(config_path, config_file_name)

    logging_config_file_name = 'logging.ini'
    logging_config_file = os.path.join(config_path, logging_config_file_name)

    config = ConfigFactory.parse_file(config_file)

    if logging_config_file:
        flat_config = convert_to_flat_dict(config)
        fileConfig(fname=logging_config_file, defaults=flat_config)

    # Preprocessing config
    preprocessing_config_dict = convert_to_dict(config['preprocessing'])
    preprocessing_config = PreprocessingConfig(preprocessing_config_dict)

    cryptonalysis_config = CryptonalysisConfig(
        crypto=config['crypto'],
        preprocessing_config=preprocessing_config
    )
    print('Cryptonalysis config:', cryptonalysis_config)
    return cryptonalysis_config


def load_training_config(config_path):
    config_file_name = 'training.conf'
    config_file = os.path.join(config_path, config_file_name)

    logging_config_file_name = 'logging.ini'
    logging_config_file = os.path.join(config_path, logging_config_file_name)

    config = ConfigFactory.parse_file(config_file)

    if logging_config_file:
        flat_config = convert_to_flat_dict(config)
        fileConfig(fname=logging_config_file, defaults=flat_config)

    config_dict = convert_to_dict(config)
    cryptonalysis_config_grid = build_cryptonalysis_config_grid(config_dict)

    return cryptonalysis_config_grid


if __name__ == '__main__':
    config1 = load_preprocessing_config('config')
    config2 = load_training_config('config')
