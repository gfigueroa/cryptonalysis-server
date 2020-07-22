# -*- coding: utf-8 -*-
"""
Elasticsearch configuration classes.
"""

from config_base import Config, load_config, convert_to_dict


class DBConfig(Config):
    # Default values
    DB = None
    HOST = 'localhost'
    USER = None
    PASSWORD = None

    def __init__(self, db_config_dict):
        """
        Initialize DBConfig object with a dictionary of values.
        :param db_config_dict
        :type db_config_dict: dict
        """
        super(DBConfig, self).__init__(db_config_dict)

        if 'port' not in db_config_dict:
            raise ValueError("Missing port in DBConfig!")

        self.db = db_config_dict['db'] = db_config_dict['db'] \
            if 'db' in db_config_dict else DBConfig.DB
        self.host = db_config_dict['host'] = db_config_dict['host'] \
            if 'host' in db_config_dict else DBConfig.HOST
        self.port = db_config_dict['port'] = db_config_dict['port']
        self.user = db_config_dict['user'] = db_config_dict['user'] \
            if 'user' in db_config_dict else DBConfig.USER
        self.password = db_config_dict['password'] = db_config_dict['password'] \
            if 'password' in db_config_dict else DBConfig.PASSWORD

        self.config_dict = db_config_dict


class IndexSettings(Config):
    # Default values
    NUMBER_OF_SHARDS = 3
    NUMBER_OF_REPLICAS = 1
    INDEX_MAPPER_DYNAMIC = False
    INDEX_RETRIES = 10  # Number of times to retry indexing a document
    INDEX_RETRY_TIMEOUT = 10  # Seconds to wait between each index retry

    def __init__(self, index_settings_config_dict):
        """
        Initialize IndexSettings object with a dictionary of values.
        :param index_settings_config_dict
        :type index_settings_config_dict: dict
        """
        super(IndexSettings, self).__init__(index_settings_config_dict)

        self.number_of_shards = index_settings_config_dict['number_of_shards'] = \
            index_settings_config_dict['number_of_shards'] if 'number_of_shards' in index_settings_config_dict \
            else IndexSettings.NUMBER_OF_SHARDS
        self.number_of_replicas = index_settings_config_dict['number_of_replicas'] = \
            index_settings_config_dict['number_of_replicas'] if 'number_of_replicas' in index_settings_config_dict \
            else IndexSettings.NUMBER_OF_REPLICAS
        self.index_mapper_dynamic = index_settings_config_dict['index_mapper_dynamic'] = \
            index_settings_config_dict['index_mapper_dynamic'] if 'index_mapper_dynamic' in index_settings_config_dict \
            else IndexSettings.INDEX_MAPPER_DYNAMIC
        self.index_retries = index_settings_config_dict['index_retries'] = \
            index_settings_config_dict['index_retries'] if 'index_retries' in index_settings_config_dict \
            else IndexSettings.INDEX_RETRIES
        self.index_retry_timeout = index_settings_config_dict['index_retry_timeout'] = \
            index_settings_config_dict['index_retry_timeout'] if 'index_retry_timeout' in index_settings_config_dict \
            else IndexSettings.INDEX_RETRY_TIMEOUT

        self.config_dict = index_settings_config_dict


class ElasticsearchConfig(Config):
    # Default values
    INDEX_SETTINGS = IndexSettings({})

    def __init__(self, elasticsearch_config_dict):
        """
        Initialize ElasticsearchConfig object with a dictionary of values.
        :param elasticsearch_config_dict
        :type elasticsearch_config_dict: dict
        """
        super(ElasticsearchConfig, self).__init__(elasticsearch_config_dict)

        self.index_settings = IndexSettings(elasticsearch_config_dict['index_settings']) \
            if 'index_settings' in elasticsearch_config_dict else ElasticsearchConfig.INDEX_SETTINGS
        elasticsearch_config_dict['index_settings'] = self.index_settings.config_dict

        self.config_dict = elasticsearch_config_dict


class ElasticsearchInjectorConfig(Config):
    # Default values
    ELASTICSEARCH_CONFIG = ElasticsearchConfig({})

    def __init__(self, elasticsearch_injector_config_dict):
        """
        Initialize ElasticsearchInjectorConfig object.
        :param elasticsearch_injector_config_dict
        :type elasticsearch_injector_config_dict: dict
        """
        super(ElasticsearchInjectorConfig, self).__init__(elasticsearch_injector_config_dict)

        if 'connections' not in elasticsearch_injector_config_dict:
            raise ValueError("Missing connections dictionary in ElasticsearchInjectorConfig!")

        self.connections = {connection: DBConfig(elasticsearch_injector_config_dict['connections'][connection])
                            for connection in elasticsearch_injector_config_dict['connections']}

        self.elasticsearch = ElasticsearchConfig(elasticsearch_injector_config_dict['elasticsearch']) \
            if 'elasticsearch' in elasticsearch_injector_config_dict \
            else ElasticsearchInjectorConfig.ELASTICSEARCH_CONFIG
        elasticsearch_injector_config_dict['elasticsearch'] = self.elasticsearch.config_dict

        self.config_dict = elasticsearch_injector_config_dict


def load_elasticsearch_injector_config(config_path, config_file_name):
    config = load_config(config_path, config_file_name)

    config_dict = convert_to_dict(config)
    elasticsearch_injector_config = ElasticsearchInjectorConfig(config_dict)

    return elasticsearch_injector_config


if __name__ == '__main__':
    eic = load_elasticsearch_injector_config('config', 'elasticsearch_injector.conf')
