"""
Helper functions for performing CRUD operations on Elasticsearch.
"""

import logging
import json
import time
import urllib3

from elasticsearch import Elasticsearch, ConnectionError, RequestError
from cryptonalysis.config import DBConfig
from cryptonalysis.utils.exceptions import ExternalUnavailabilityError

# TODO: Add certificate verification?
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger()

__elasticsearch = None  # Ensure singleton for Elasticsearch class
__current_index = None  # Optimize index creation

MAX_INDEX_RETRIES = 100  # Ensure a maximum number of index retries


class ElasticsearchInjector(object):
    def __init__(self, config):
        """
        Initialize the ElasticsearchInjector.
        :param config: configuration object
        :type config: ElasticsearchInjectorConfig
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Elasticsearch
        es_db_config = config.connections['elasticsearch']
        self.elasticsearch = get_es_connection(es_db_config)

    def dump(self, document, document_id, index_name, mappings):
        """
        Dump (store) a given object (dictionary) into Elasticsearch.
        :param document: A dictionary representing an Elasticsearch document to be stored
        :type document: dict
        :param document_id: The ID of the Elasticsearch document to store
        :type document_id: str
        :param index_name: The name of the ES index where the document will be indexed
        :type index_name: str
        :param mappings: A dictionary of Elasticsearch mappings for the document to store
        :type mappings: dict
        """
        es_document = json.dumps(document, sort_keys=True)
        if not store_es_document(self.elasticsearch, self.config.elasticsearch, es_document, index_name, mappings,
                                 document_id):
            raise ExternalUnavailabilityError('Error indexing document to Elasticsearch')


def get_es_connection(es_db_config):
    """
    Connect to Elasticsearch and return an instance of Elasticsearch class.
    :param es_db_config: Configuration for Elasticsearch connection
    :type es_db_config: DBConfig
    :return: An instance of the Elasticsearch class
    :rtype: Elasticsearch
    """
    global __elasticsearch
    if __elasticsearch is not None:
        return __elasticsearch

    use_ssl = es_db_config.port == 443 or es_db_config.port == 8443
    elasticsearch_config = {
        'host': es_db_config.host,
        'port': es_db_config.port,
        'use_ssl': use_ssl,
        'verify_certs': False
    }
    if not use_ssl:
        elasticsearch_config['http_auth'] = (es_db_config.user, es_db_config.password)

    logger.info('Connecting to Elasticsearch...')
    elasticsearch = Elasticsearch([elasticsearch_config])
    if not elasticsearch.ping():
        raise ExternalUnavailabilityError('Could not establish connection with Elasticsearch')

    __elasticsearch = elasticsearch
    logger.info('Connection to Elasticsearch established')

    return __elasticsearch


def create_es_index(elasticsearch, es_config, index_name, mappings):
    """
    Creates an index in Elasticsearch.
    :param elasticsearch
    :param es_config
    :param index_name
    :param mappings: A mappings dictionary for the Elasticsearch index
    :return: True if the index was created, False otherwise.
    """
    # index settings
    settings = {
        'settings': {
            'number_of_shards': es_config.index_settings.number_of_shards,
            'number_of_replicas': es_config.index_settings.number_of_replicas,
            'index.mapper.dynamic': es_config.index_settings.index_mapper_dynamic
        },
        'mappings': mappings
    }

    try:
        # Ignore 400 means to ignore "Index Already Exists" error.
        result = elasticsearch.indices.create(index=index_name, ignore=400, body=settings)

        # Assign index_name to __current_index to avoid repeated calls to ES
        global __current_index
        __current_index = index_name

        if 'error' not in result:
            logger.info("Created Elasticsearch index '{}'".format(index_name))
            return True
        else:
            return False
    except Exception as e:
        logger.error("Error creating Elasticsearch index '{}': {}".format(index_name, e))
        raise e


def update_mappings(elasticsearch, es_config, index_name, mappings):
    """
    Update the mappings of a given index.
    :param elasticsearch
    :param es_config
    :param index_name: The index name
    :param mappings: A mappings dictionary for the Elasticsearch index
    """
    retries = 0
    while retries < MAX_INDEX_RETRIES:
        try:
            elasticsearch.indices.put_mapping(body=mappings, index=index_name, doc_type='_doc')
            break
        except ConnectionError as e:
            logger.error("Error connecting to Elasticsearch.", exc_info=e)

            if retries < es_config.index_settings.index_retries:
                logger.info("Retrying Elasticsearch mapping update in {} seconds...".format(
                    es_config.index_settings.index_retry_timeout))
                retries += 1
                time.sleep(es_config.index_settings.index_retry_timeout)
            else:
                raise e
        except Exception as e:
            logger.error("Error updating mappings.", exc_info=True)
            raise e


def store_es_document(elasticsearch, es_config, document, index_name, mappings,
                      doc_id=None):
    """
    Stores an Elasticsearch document to a given index.
    :param elasticsearch
    :param es_config
    :param document: The document to store in Json format
    :param index_name: The index name
    :param mappings: A mappings dictionary for the Elasticsearch index
    :param doc_id: The id of the document(optional)
    :return: True if document was stored successfully, False otherwise.
    """
    retries = 0
    while retries < MAX_INDEX_RETRIES:
        try:
            # Create index if necessary
            if __current_index != index_name:
                create_es_index(elasticsearch, es_config, index_name, mappings)

            # Index the document
            logger.debug('Dumping document to Elasticsearch: ' + document)
            elasticsearch.index(index=index_name, body=document, id=doc_id)
            return True
        except ConnectionError as e:
            logger.error("Error connecting to Elasticsearch.", exc_info=e)

            if retries < es_config.index_settings.index_retries:
                logger.info("Retrying Elasticsearch indexing in {} seconds...".format(
                    es_config.index_settings.index_retry_timeout))
                retries += 1
                time.sleep(es_config.index_settings.index_retry_timeout)
            else:
                return False
        except RequestError as e:
            # Handle new mapping in code (for forward compatibility)
            if e.error == 'strict_dynamic_mapping_exception':
                if verify_mappings(json.loads(document), mappings):
                    logger.info('New fields found in index definition.')
                    update_mappings(elasticsearch, es_config, index_name, mappings)
                    continue
            raise e
        except Exception as e:
            logger.error("Error storing Elasticsearch document.", exc_info=True)
            raise e

    return False


def get_es_document(elasticsearch, es_config, index_name, doc_id):
    """
    Fetches an Elasticsearch document from a given index.
    :param elasticsearch
    :param es_config
    :param index_name: The index name
    :param doc_id: The id of the document
    :return: a document
    """
    retries = 0
    while retries < MAX_INDEX_RETRIES:
        try:
            res = elasticsearch.get(index=index_name, id=doc_id)
            return res['_source']
        except ConnectionError as e:
            logger.error("Error connecting to Elasticsearch.", exc_info=True)

            if retries < es_config.index_settings.index_retries:
                logger.info("Retrying Elasticsearch indexing in {} seconds...".format(
                    es_config.index_settings.index_retry_timeout))
                retries += 1
                time.sleep(es_config.index_settings.index_retry_timeout)
            else:
                raise e
        except Exception as e:
            logger.error("Error getting Elasticsearch document.", exc_info=True)
            raise e


def verify_mappings(document, mappings):
    """
    Test that an Elasticsearch document to contains the fields specified in a given mappings dictionary.
    Some fields from the specified mappings may not be present in the document, but all fields in the document must
    appear in the mappings.
    The method only checks for first and second level mappings (i.e. nested objects inside nested objects are not taken
    into account.
    :param document
    :param mappings
    :return: True if the Elasticsearch document agrees with the given mappings, False otherwise
    """
    for attribute_name in document:
        attribute_value = document[attribute_name]
        mappings_type = '_doc'
        mapping_properties = mappings[mappings_type]['properties']

        # Check property exists
        if attribute_name not in mapping_properties.keys():
            return False

        # Check nested properties
        if mapping_properties[attribute_name]['type'] == 'nested':
            nested_mapping_properties = mapping_properties[attribute_name]['properties']
            if type(attribute_value) is list and len(attribute_value):
                nested_attribute = attribute_value[0]
                for nested_attribute_name in nested_attribute:
                    # Check property exists
                    if nested_attribute_name not in nested_mapping_properties:
                        return False

    return True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
