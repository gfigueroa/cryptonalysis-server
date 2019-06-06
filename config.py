# -*- coding: utf-8 -*-
"""
Configuration classes.
"""

import os
from datetime import date
from logging.config import fileConfig
from pyhocon import ConfigFactory
from cryptonalysis.utils import misc_utils
from cryptonalysis.ml_core.transaction_builders import get_predictor_class_from_name


class PreprocessingConfig(object):
    def __init__(self, start_date, end_date, price_column, save_roi, save_data, predictor_cls, predictor_params):
        """
        Initialize PreprocessingConfig object.
        :param start_date
        :type start_date: date
        :param end_date
        :type end_date: date
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
        self.price_column = price_column
        self.save_roi = save_roi
        self.save_data = save_data
        self.predictor_cls = predictor_cls
        self.predictor_params = predictor_params


class CryptonalysisConfig(object):

    def __init__(self, crypto, preprocessing_config):
        """
        Initialize ClassificationConfig object.
        :param crypto: The cryptocurrency to use (e..g ETH, BTC, etc.)
        :type crypto: str
        :param preprocessing_config
        :type preprocessing_config: PreprocessingConfig
        """
        self.crypto = crypto
        self.preprocessing_config = preprocessing_config


def convert_to_flat_dict(conf, prefix=""):
    result = {}
    for k in conf:
        v = conf[k]
        if isinstance(v, dict):
            for k1 in convert_to_flat_dict(v, k + "."):
                result[prefix + k1] = conf[k1]
        else:
            result[prefix + k] = v
    return result


def load_config(config_path):
    env = os.environ['ENVIRONMENT'] if 'ENVIRONMENT' in os.environ else 'local'

    config_file_name = "{}.conf".format(env)
    config_file = os.path.join(config_path, config_file_name)

    logging_config_file_name = 'logging.ini'
    logging_config_file = os.path.join(config_path, logging_config_file_name)

    config = ConfigFactory.parse_file(config_file)

    if logging_config_file:
        flat_config = convert_to_flat_dict(config)
        fileConfig(fname=logging_config_file, defaults=flat_config)

    start_date = misc_utils.parse_date(config['preprocessing']['start_date'])
    end_date = misc_utils.parse_date(config['preprocessing']['end_date'])
    predictor_cls = get_predictor_class_from_name(config['preprocessing']['predictor_cls'])
    predictor_params = {
        k: config['preprocessing']['predictor_params'][k]
        for k in config['preprocessing']['predictor_params']
    }
    cryptonalysis_config = CryptonalysisConfig(
        crypto=config['crypto'],
        preprocessing_config=PreprocessingConfig(
            start_date=start_date,
            end_date=end_date,
            price_column=config['preprocessing']['price_column'],
            save_roi=config['preprocessing']['save_roi'],
            save_data=config['preprocessing']['save_data'],
            predictor_cls=predictor_cls,
            predictor_params=predictor_params
        )
    )
    print('Cryptonalysis config:', cryptonalysis_config)
    return cryptonalysis_config
