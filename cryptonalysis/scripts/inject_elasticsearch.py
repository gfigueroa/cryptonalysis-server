"""
Inject various data into elasticsearch.
"""

import logging
import os

from cryptonalysis.config.elasticsearch_config import load_elasticsearch_injector_config
from cryptonalysis.ml_core.market import MarketParameters
from cryptonalysis.ml_core.prediction import run_multiple_simulations
from cryptonalysis.ml_core.transaction_builders import get_transaction_type
from cryptonalysis.utils.elasticsearch_helpers import ElasticsearchInjector


TRANSACTION_SIMULATION_INDEX = 'transaction_simulations'
TRANSACTION_SIMULATION_MAPPINGS = {
    '_doc': {
        'dynamic': 'strict',
        'properties': {
            'configuration_file': {'type': 'keyword'},
            'crypto': {'type': 'keyword'},
            'current_price': {'type': 'double'},
            'deep_learning_activation_function': {'type': 'keyword'},
            'deep_learning_neurons': {'type': 'integer'},
            'market_min_transaction_size_crypto': {'type': 'double'},
            'market_min_transaction_size_fiat': {'type': 'double'},
            'market_max_transaction_size_crypto': {'type': 'double'},
            'market_max_transaction_size_fiat': {'type': 'double'},
            'preprocessing_end_date': {'type': 'date'},
            'preprocessing_normalize': {'type': 'boolean'},
            'preprocessing_normalize_by_row': {'type': 'boolean'},
            'preprocessing_predictor_cls': {'type': 'keyword'},
            'preprocessing_predictor_params_daily_allowance': {'type': 'integer'},
            'preprocessing_predictor_params_lookahead_days': {'type': 'integer'},
            'preprocessing_predictor_params_prob_buy': {'type': 'float'},
            'preprocessing_predictor_params_prob_sell': {'type': 'float'},
            'preprocessing_predictor_params_starting_investment': {'type': 'integer'},
            'preprocessing_price_column': {'type': 'keyword'},
            'preprocessing_standardize': {'type': 'boolean'},
            'preprocessing_start_date': {'type': 'date'},
            'preprocessing_window_size': {'type': 'integer'},
            'training_cv_folds': {'type': 'integer'},
            'training_dev_size': {'type': 'float'},
            'training_shuffle_data': {'type': 'boolean'},
            'training_training_size': {'type': 'float'},
            'results_acc': {'type': 'double'},
            'results_capital': {'type': 'double'},
            'results_cash': {'type': 'double'},
            'results_date': {'type': 'date'},
            'results_model': {'type': 'keyword'},
            'results_owned_crypto': {'type': 'double'},
            'results_y_pred': {'type': 'keyword'},
            'results_y_true': {'type': 'keyword'},
            'results_roi': {'type': 'double'},
            'results_roi_perc': {'type': 'double'},
            'results_total_investment': {'type': 'double'}
        }
    }
}


# Configuration params to remove from Elasticsearch document
IRRELEVANT_CONFIGURATION_PARAMS = [
    'logging_handlers',
    'logging_level',
    'save_preprocessing_data',
    'save_preprocessing_roi',
    'save_training_model',
    'save_training_results'
]


# Override investment parameters
STARTING_INVESTMENT = [100]
DAILY_ALLOWANCE = [5]
MARKET_PARAMETERS = {
    'min_transaction_size_crypto': [0.01, 0.1],
    'min_transaction_size_fiat': [1, 10],
    'max_transaction_size_crypto': [100, 1000],
    'max_transaction_size_fiat': [100, 1000]
}


def _remove_irrelevant_configuration_params(conf):
    """
    Remove configuration params that will not be saved in Elasticsearch document.
    :param conf: flat configuration
    :type conf: dict
    :return: an updated configuration
    :rtype: dict
    """
    for param in IRRELEVANT_CONFIGURATION_PARAMS:
        if param in conf.keys():
            conf.pop(param)

    return conf


def _market_parameter_combinations():
    market_parameter_combinations = []
    for min_transaction_size_crypto in MARKET_PARAMETERS['min_transaction_size_crypto']:
        for min_transaction_size_fiat in MARKET_PARAMETERS['min_transaction_size_fiat']:
            for max_transaction_size_crypto in MARKET_PARAMETERS['max_transaction_size_crypto']:
                for max_transaction_size_fiat in MARKET_PARAMETERS['max_transaction_size_fiat']:
                    market_parameters = MarketParameters(min_transaction_size_crypto=min_transaction_size_crypto,
                                                         min_transaction_size_fiat=min_transaction_size_fiat,
                                                         max_transaction_size_crypto=max_transaction_size_crypto,
                                                         max_transaction_size_fiat=max_transaction_size_fiat)
                    market_parameter_combinations.append(market_parameters)
    return market_parameter_combinations


def _market_parameters_string(market_parameters):
    """
    Get a single-line string from the market parameters used.
    :param market_parameters
    :type market_parameters: MarketParameters
    :return: a single-line string
    :rtype: str
    """
    return "{}{}{}{}".format(market_parameters.min_transaction_size_crypto, market_parameters.min_transaction_size_fiat,
                             market_parameters.max_transaction_size_crypto, market_parameters.max_transaction_size_fiat)


def _get_document_id(conf, market_parameters, starting_investment, daily_allowance, model, date_str):
    return "{}_{}_{}_{}_{}_{}".format(conf.to_single_line_str(), _market_parameters_string(market_parameters),
                                      str(starting_investment), str(daily_allowance), model, date_str)


def inject_transaction_simulation_data(config_path):
    injector_config = load_elasticsearch_injector_config('config', 'elasticsearch_injector.conf')
    injector = ElasticsearchInjector(injector_config)

    for market_parameters in _market_parameter_combinations():
        for starting_inv in STARTING_INVESTMENT:
            for daily_allowance in DAILY_ALLOWANCE:
                for conf_file, conf, results, y_true in run_multiple_simulations(config_path,
                                                                                 market_parameters=market_parameters,
                                                                                 starting_investment=starting_inv,
                                                                                 daily_allowance=daily_allowance):
                    flat_conf = conf.to_serializable_dict(separator='_')
                    flat_conf = _remove_irrelevant_configuration_params(flat_conf)

                    if 'deep_learning' in flat_conf and flat_conf['deep_learning'] is None:
                        flat_conf.pop('deep_learning')
                        flat_conf['deep_learning_activation_function'] = None
                        flat_conf['deep_learning_neurons'] = None

                    # Configuration fields
                    document = {
                        'configuration_file': conf_file
                    }
                    document.update(flat_conf)

                    # Market fields
                    document['market_min_transaction_size_crypto'] = market_parameters.min_transaction_size_crypto
                    document['market_min_transaction_size_fiat'] = market_parameters.min_transaction_size_fiat
                    document['market_max_transaction_size_crypto'] = market_parameters.max_transaction_size_crypto
                    document['market_max_transaction_size_fiat'] = market_parameters.max_transaction_size_fiat

                    # Results fields
                    for model, results_dict in results.items():
                        document['results_model'] = model
                        document['results_acc'] = results_dict['acc']
                        for index, row in results_dict['predictor_states'].iterrows():
                            document['current_price'] = row['price']
                            document['results_capital'] = row['capital']
                            document['results_cash'] = row['cash']
                            document['results_date'] = index.strftime('%Y-%m-%d')
                            document['results_owned_crypto'] = row['owned_crypto']
                            document['results_y_pred'] = get_transaction_type(results_dict['y_pred'][index])
                            document['results_y_true'] = get_transaction_type(y_true[index])
                            document['results_roi'] = row['roi']
                            document['results_roi_perc'] = row['roi_perc']
                            document['results_total_investment'] = row['total_investment']

                            date_str = index.strftime('%Y-%m-%d')
                            document_id = _get_document_id(conf, market_parameters, starting_inv, daily_allowance,
                                                           model, date_str)
                            injector.dump(document, document_id, TRANSACTION_SIMULATION_INDEX,
                                          TRANSACTION_SIMULATION_MAPPINGS)


if __name__ == '__main__':
    conf_path = 'config'
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))  # Necessary for initial logs
    inject_transaction_simulation_data(conf_path)
