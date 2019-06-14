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
    def __init__(self, start_date, end_date, window_size, normalize, normalize_by_row, price_column, save_roi,
                 save_data, predictor_cls, predictor_params):
        """
        Initialize PreprocessingConfig object.
        :param start_date
        :type start_date: date
        :param end_date
        :type end_date: date
        :param window_size
        :type window_size: int
        :param normalize
        :type normalize: bool
        :param normalize_by_row
        :type normalize_by_row: bool
        :param price_column
        :type price_column: str
        :param save_roi
        :type save_roi: bool
        :param save_data
        :type save_data: bool
        :param predictor_cls
        :type predictor_cls: ``classobj``
        :param predictor_params
        :type predictor_params: dict
        """
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.normalize = normalize
        self.normalize_by_row = normalize_by_row
        self.price_column = price_column
        self.save_roi = save_roi
        self.save_data = save_data
        self.predictor_cls = predictor_cls
        self.predictor_params = predictor_params


class TrainingConfig(object):

    def __init__(self, preprocessing_config, shuffle_data):
        self.preprocessing_config = preprocessing_config
        self.shuffle_data = shuffle_data


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
            for k1 in convert_to_dict(v):
                result[k1] = config[k1]
        else:
            result[k] = v
    return result


def build_cryptonalysis_config_grid(config, grid=None):
    """

    :param config:
    :type config: dict
    :param grid:
    :return:
    """
    if grid is None:
        grid = []

    for param in config:
        if type(config[param]) is list:
            for value in config[param]:
                temp_config = config.copy()[param] = value
                grid.append(build_cryptonalysis_config_grid(temp_config, grid))
        else:
            grid.append(config)

    return grid


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
    start_date = misc_utils.parse_date(config['preprocessing']['start_date'])
    end_date = misc_utils.parse_date(config['preprocessing']['end_date'])
    predictor_cls = get_predictor_class_from_name(config['preprocessing']['predictor_cls'])
    preprocessing_config = PreprocessingConfig(
        start_date=start_date,
        end_date=end_date,
        window_size=config['preprocessing']['window_size'],
        normalize=config['preprocessing']['normalize'],
        normalize_by_row=config['preprocessing']['normalize_by_row'],
        price_column=config['preprocessing']['price_column'],
        save_roi=config['preprocessing']['save_roi'],
        save_data=config['preprocessing']['save_data'],
        predictor_cls=predictor_cls,
        predictor_params=convert_to_dict(config['preprocessing']['predictor_params'])
    )

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

    cryptonalysis_config_grid = build_cryptonalysis_config_grid(convert_to_dict(config))

    return cryptonalysis_config_grid


if __name__ == '__main__':
    config1 = load_preprocessing_config('config')
    config2 = load_training_config('config')
