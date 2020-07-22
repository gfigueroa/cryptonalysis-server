# -*- coding: utf-8 -*-
"""
Various functions and classes for configuration management.
"""

import json
import os
from datetime import date
from logging.config import fileConfig
from pandas.io.json import json_normalize
from pyhocon import ConfigFactory, ConfigTree


class Config(object):

    def __init__(self, config_dict):
        self.config_dict = config_dict

    def to_csv_str(self):
        flat_config_dict = json_normalize(self.config_dict).to_dict(orient='records')[0]
        sorted_keys = sorted(flat_config_dict.keys())
        sorted_keys_str = ','.join(sorted_keys)
        values = ','.join([str(flat_config_dict[k]
                               if type(flat_config_dict[k]) is not date else flat_config_dict[k].strftime('%Y-%m-%d'))
                           for k in sorted_keys])
        return sorted_keys_str, values

    def to_single_line_str(self):
        flat_config_dict = json_normalize(self.config_dict).to_dict(orient='records')[0]
        sorted_keys = sorted(flat_config_dict.keys())
        values = ''.join([str(flat_config_dict[k]
                              if type(flat_config_dict[k]) is not date else flat_config_dict[k].strftime('%Y-%m-%d'))
                          for k in sorted_keys])
        return values

    def to_serializable_dict(self, separator='.'):
        flat_config_dict = json_normalize(self.config_dict, sep=separator).to_dict(orient='records')[0]
        return {
            k: v if type(v) is not date else v.strftime('%Y-%m-%d')
            for k, v in flat_config_dict.items()
        }

    def __str__(self):
        return json.dumps(self.to_serializable_dict(), indent=2)


def convert_to_flat_dict(config, prefix="", separator="."):
    """
    Recursively convert ConfigTree or dictionary into a flat dictionary.
    :param config
    :type config: ConfigTree or dict
    :param prefix: Prefix to add to each key when flattening
    :param separator: Separator to use for flattening
    :return: A flat configuration tree
    :rtype: dict
    """
    result = {}
    for k in config:
        v = config[k]
        if isinstance(v, dict):
            for k1 in convert_to_flat_dict(v, k + separator):
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


def load_config(config_path, config_file_name):
    config_file = os.path.join(config_path, config_file_name)

    logging_config_file_name = 'logging.ini'
    logging_config_file = os.path.join(config_path, logging_config_file_name)

    config = ConfigFactory.parse_file(config_file)

    if logging_config_file:
        flat_config = convert_to_flat_dict(config)
        fileConfig(fname=logging_config_file, defaults=flat_config)

    return config
