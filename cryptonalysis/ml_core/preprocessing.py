import logging
import os
import pandas as pd
import sys
from cryptonalysis.config import load_cryptonalysis_config_grid, PreprocessingConfig, CryptonalysisConfigGrid
from joblib import dump, load
from market import market
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transaction_builders import TRANSACTION_TYPE

# Logging
logger = logging.getLogger()

# Constants
MASTER_DATA_DIR = os.path.join('cryptonalysis', 'data', 'master_data')
PREPROCESSED_DATA_DIR = os.path.join('cryptonalysis', 'data', 'preprocessed')
CRYPTOCURRENCIES = {
    'ETH': "ethereum",
    'BTC': "bitcoin",
    'XRP': "ripple",
    'LTC': "litecoin",
    'USDT': "tether"
}


def get_historical_df(historical_file):
    """
    Gets a new DataFrame containing the historical data for a particular cryptocurrency.
    :param historical_file: The file path containing the historical data
    :return: DataFrame instance
    :rtype: DataFrame
    """
    logger.info("Loading historical data from {}...".format(historical_file))

    df = pd.read_csv(historical_file, sep='\t', thousands=',')

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Cleanup
    if df['Market Cap'].dtype == 'object':
        # Remove '-'
        df.loc[df['Market Cap'] == '-', 'Market Cap'] = 0

        # Convert to numeric
        df['Market Cap'] = df['Market Cap'].str.replace(',', '')
        df['Market Cap'] = pd.to_numeric(df['Market Cap'])
        df = df.fillna(0)

    # Set index
    df = df.set_index('Date').loc[:, 'Open':]

    return df


def get_price_list(df, price_column):
    """
    Get a DataFrame with a single column of prices given a composite DataFrame and the name of the price column to
    consider.
    :param df: The original cryptocurrency DataFrame.
    :type df: DataFrame
    :param price_column: The name of the column containing the crypto price to consider.
    :type price_column: str
    :return: A Series containing cryptocurrency prices
    :rtype: Series
    """
    # Get transaction data
    price_list = df[price_column]
    return price_list


def build_transactions_df(transactions):
    """
    Build a DataFrame ready for classification (i.e., X and y) based on a dictionary of transactions.
    :param transactions: A list of dictionaries of the form
    [{'transaction': 'TRANSACTION_TYPE', 'prices': [price1, price2, ...]}, ...]
    :type transactions: list of dict
    :return: A DataFrame with N price columns and a transaction column (1=BUY, 0=SELL, -1=UNKNOWN)
    :rtype: DataFrame
    """

    logger.info("Getting transactions DataFrame...")

    # Price columns
    prices = list(transactions[0]['prices'])  # Get first list of prices to obtain length
    df_columns = ["price {0}".format(column_name) for column_name in range(1, len(prices) + 1)]
    df_columns.append('transaction')

    df = DataFrame(columns=df_columns)
    for row in range(len(transactions)):
        prices = list(transactions[row]['prices'])
        transaction = TRANSACTION_TYPE[transactions[row]['transaction']]
        df.loc[row] = prices + [transaction]

    date_index = [t['prices'].index[-1] for t in transactions]  # Use last date of each transaction
    df.index = pd.DatetimeIndex(date_index)
    df['transaction'] = df['transaction'].astype(int)

    return df


def normalize_df(df, by_row, standardize=False):
    """
    Normalize a DataFrame either by row (sample per sample) or by column (feature scaling). It can either use
    standardization (StandardScaler) or min max normalization (MinMaxScaler).
    Standardization uses the standard score of a sample:
    x' = (x - u) / s, where u is the mean and s is the standard deviation
    Normalization uses the min nax scaling formula to be in the range [0, 1]:
    x' = (x - x_min) / (x_max - x_min)
    When normalizing by column, the scaler is also returned by the function, since the same scaler is required for
    training, testing, and prediction. The scaler becomes useless after normalization when normalizing by row.
    :param df: The DataFrame to normalize
    :type df: DataFrame
    :param by_row: If True, normalizes by row using, taking each sample independently, otherwise it normalizes by column
    :type by_row: bool
    :param standardize: Whether to use StandardScaler (if True) or MinMaxScaler (if False)
    :type standardize: bool
    :return: A tuple with the normalized Dataframe and the scaler (or None if by_row=True)
    :rtype: (DataFrame, StandardScaler or MinMaxScaler or None)
    """

    logger.info("Normalizing data...")

    if standardize:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    norm_df = df.copy(deep=True)  # type: DataFrame
    only_data = norm_df.iloc[:, :-1]
    if by_row:
        scaler = scaler.fit(only_data.transpose())
        norm_df.iloc[:, :-1] = scaler.transform(only_data.transpose()).transpose()
    else:
        scaler = scaler.fit(only_data)
        norm_df.iloc[:, :-1] = scaler.transform(only_data)

    # Return scaler only if not by_row
    if by_row:
        scaler = None

    return norm_df, scaler


def normalize_df_with_scaler(df, scaler):
    """
    Normalize a DataFrame by column using a given fitted scaler (e.g. MinMaxScaler, StandardScaler). This method is to
    be called in data for testing, simulations and predictions. The scaler must previously have been fitted with
    training data, or a ValueError is raised.
    :param df: The DataFrame to normalize
    :type df: DataFrame
    :param scaler: A fitted scaler instance
    :type scaler: StandardScaler or MinMaxScaler
    :return: The normalized DataFrame
    :rtype: DataFrame
    """

    logger.info("Normalizing data...")
    if not hasattr(scaler, 'n_samples_seen_') or scaler.n_samples_seen_ < 1:
        raise ValueError("Scaler has not been fitted with training data!")

    norm_df = df.copy(deep=True)  # type: DataFrame
    norm_df.iloc[:, :-1] = scaler.transform(norm_df.iloc[:, :-1])

    return norm_df


def normalize_df_with_scaler_params(df, crypto_name, preprocessing_config, scaler_dir=None):
    """
    Normalize a DataFrame by column using a given the preprocessing parameters of a previously saved and fitted scaler.
    :param df: The DataFrame to normalize
    :type df: DataFrame
    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param scaler_dir: (Default None) The (overridden) directory where the scaler is saved. If None, the default
    `PREPROCESSED_DATA_DIR` is used.
    :type scaler_dir: str
    :return: The normalized DataFrame
    :rtype: DataFrame
    """
    scaler = load_scaler(crypto_name, preprocessing_config, scaler_dir)
    return normalize_df_with_scaler(df, scaler)


def get_data_filename(crypto_name, preprocessing_config):
    """
    Get the path and filename used for the data file containing the preprocessed data based on the preprocessing
    parameters.

    >>> get_data_filename('ETH',
    ...     PreprocessingConfig(
    ...     {
    ...         'window_size': 60,
    ...         'predictor_params': {
    ...             'lookahead_days': 3, 'prob_buy': 1, 'prob_sell': 1, 'starting_investment': 100, 'daily_allowance': 5
    ...         },
    ...         'start_date': '2018-01-01',
    ...         'end_date': '2019-05-04',
    ...         'price_column': 'Close',
    ...         'normalize': True,
    ...         'normalize_by_row': True,
    ...         'standardize': False,
    ...         'predictor_cls': 'BiffPredictor',
    ...     })
    ... )
    'pre_ETH_2019-05-04TrueTrueBiffPredictor5311100CloseFalse2018-01-0160.csv'

    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :return A string with the name of a preprocessed data file given a series of config parameters
    :rtype: str
    """
    filename = 'pre_{}_{}.csv'.format(crypto_name, preprocessing_config.to_single_line_str())
    return filename


def load_preprocessed_data(crypto_name, preprocessing_config, preprocessed_data_dir=None):
    """
    Load the DataFrame (if saved) as a CSV containing data ready for the classification task.
    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param preprocessed_data_dir: (Default None) The (overridden) directory where the preprocessed data is located. If
    None, the default `PREPROCESSED_DATA_DIR` is used.
    :type preprocessed_data_dir: str
    :return: a DataFrame
    :rtype: DataFrame
    """
    data_filename = get_data_filename(crypto_name, preprocessing_config)
    dir_to_use = PREPROCESSED_DATA_DIR if preprocessed_data_dir is None else preprocessed_data_dir
    data_file_path = os.path.join(dir_to_use, data_filename)
    if os.path.isfile(data_file_path):
        logger.info("Preprocessed datafile '{0}'' already exists. "
                    "Loading file and skipping preprocessing pipeline...".format(data_file_path))
        transactions_df = pd.read_csv(data_file_path, index_col='date')
        transactions_df.index = pd.to_datetime(transactions_df.index)
        return transactions_df
    else:
        return None


def save_preprocessed_data(transactions_df, crypto_name, preprocessing_config, preprocessed_data_dir=None):
    """
    Save the DataFrame containing data ready for the classification task as a CSV file.
    :param transactions_df: The DataFrame containing the actual preprocessed data
    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param preprocessed_data_dir: (Default None) The (overridden) directory where the preprocessed data should be saved.
    If None, the default `PREPROCESSED_DATA_DIR` is used.
    :type preprocessed_data_dir: str
    """

    dir_to_use = PREPROCESSED_DATA_DIR if preprocessed_data_dir is None else preprocessed_data_dir
    if dir_to_use == PREPROCESSED_DATA_DIR and not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)

    data_filename = get_data_filename(crypto_name, preprocessing_config)
    data_file_path = os.path.join(dir_to_use, data_filename)

    logger.info("Saving data to file '{0}'".format(data_file_path))
    transactions_df.to_csv(data_file_path, index=True, index_label='date')


def save_scaler(scaler, crypto_name, preprocessing_config, scaler_dir=None):
    """
    Save a scaler obtained from normalizing/standardazing data in the preprocessing stage. This scaler can later be used
    for prediction.
    Normally, only column-based normalization/standardization will have use for a scaler.
    :param scaler: A fitted scaler instance
    :type scaler: StandardScaler or MinMaxScaler
    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param scaler_dir: (Default None) The (overridden) directory where the scaler should be saved. If None, the default
    `PREPROCESSED_DATA_DIR` is used.
    :type scaler_dir: str
    """

    if not preprocessing_config.normalize or preprocessing_config.normalize_by_row:
        raise ValueError("A scaler should not be defined for non-normalized/non-standardized data or data that has "
                         "been normalize/standardized by row.")

    dir_to_use = PREPROCESSED_DATA_DIR if scaler_dir is None else scaler_dir

    if dir_to_use == PREPROCESSED_DATA_DIR and not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)

    scaler_filename = get_data_filename(crypto_name, preprocessing_config)
    scaler_filename = scaler_filename.replace('pre_', 'scaler_')
    scaler_filename = scaler_filename.replace('.csv', '.joblib')
    scaler_file_path = os.path.join(dir_to_use, scaler_filename)
    logger.info("Saving scaler to file '{0}'".format(scaler_file_path))
    dump(scaler, scaler_file_path)


def load_scaler(crypto_name, preprocessing_config, scaler_dir=None):
    """
    Load a scaler obtained from normalizing/standardizing data in the preprocessing stage. This scaler can be used for
    prediction.
    Normally, only column-based normalization/standardization will have use for a scaler.
    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param scaler_dir: (Default None) The (overridden) directory where the scaler is saved. If None, the default
    `PREPROCESSED_DATA_DIR` is used.
    :type scaler_dir: str
    :return a fitted scaler
    :rtype: StandardScaler or MinMaxScaler
    """

    if not preprocessing_config.normalize or preprocessing_config.normalize_by_row:
        raise ValueError("A scaler should not be defined for non-normalized/non-standardized data or data that has "
                         "been normalize/standardized by row.")

    dir_to_use = PREPROCESSED_DATA_DIR if scaler_dir is None else scaler_dir

    if dir_to_use == PREPROCESSED_DATA_DIR and not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)

    scaler_filename = get_data_filename(crypto_name, preprocessing_config)
    scaler_filename = scaler_filename.replace('pre_', 'scaler_')
    scaler_filename = scaler_filename.replace('.csv', '.joblib')
    scaler_file_path = os.path.join(dir_to_use, scaler_filename)
    logger.info("Loading scaler from file '{0}'".format(scaler_file_path))

    try:
        scaler = load(scaler_file_path)
    except IOError:
        raise ValueError("Scaler with path '{}' not found!".format(scaler_file_path))

    return scaler


def preprocess_dataframe(df, crypto_name, preprocessing_config, save_roi=False, predicting=False, scaler_dir=None):
    """
    Preprocess a dataframe containing cryptocurrency information and have it ready for training/testing.
    The preprocessing runs a `CryptoPredictor`, which is necessary for training and testing models.
    :param df: The DataFrame containing the unprocessed cryptocurrency data
    :type df: DataFrame
    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param save_roi: Whether or not to save the ROI of this predictor run in a file
    :type save_roi: bool
    :param predicting: (Default False) Whether or not the preprocessing is being run for prediction (requires previously
    fitted scaler if normalization/standardization by column).
    :type predicting: bool
    :param scaler_dir: (Default None) The (overridden) directory where the scaler should be saved. If None, the default
    `PREPROCESSED_DATA_DIR` is used.
    :type scaler_dir: str
    :return: The preprocessing DataFrame containing daily transactions in the last column.
    :rtype: DataFrame
    """

    price_column = preprocessing_config.price_column
    price_list = get_price_list(df, price_column)

    # Check if dates are outside price_list time range
    preprocessing_config.adjust_dates(df.index)

    # Preprocessing parameters
    predictor_cls = preprocessing_config.predictor_cls
    window_size = preprocessing_config.window_size
    normalize = preprocessing_config.normalize
    normalize_by_row = preprocessing_config.normalize_by_row
    standardize = preprocessing_config.standardize

    # Predictor parameters
    lookahead_days = preprocessing_config.predictor_params['lookahead_days']
    starting_investment = preprocessing_config.predictor_params['starting_investment']
    daily_allowance = preprocessing_config.predictor_params['daily_allowance']
    prob_buy = preprocessing_config.predictor_params['prob_buy']
    prob_sell = preprocessing_config.predictor_params['prob_sell']

    # Run transaction builder
    predictor = predictor_cls(market, price_list, window_size, crypto_name, preprocessing_config.start_date,
                              preprocessing_config.end_date, starting_investment=starting_investment,
                              daily_allowance=daily_allowance, lookahead_days=lookahead_days,
                              prob_buy=prob_buy, prob_sell=prob_sell)
    predictor.run_predictor(save_roi)

    # Get transactions DataFrame
    transactions_df = build_transactions_df(predictor.transactions)

    logger.info('Buy: {0}'.format(len(transactions_df[transactions_df['transaction'] == TRANSACTION_TYPE['BUY']])))
    logger.info('Sell: {0}'.format(len(transactions_df[transactions_df['transaction'] == TRANSACTION_TYPE['SELL']])))

    # Data normalization
    # TODO: Normalization should be part of training stage
    if normalize:
        if not predicting:
            transactions_df, scaler = normalize_df(transactions_df, normalize_by_row, standardize)
            # Only save scaler if normalization/standardization is done by column
            if not normalize_by_row:
                save_scaler(scaler, crypto_name, preprocessing_config, scaler_dir)
        else:  # For prediction simulations
            if normalize_by_row:
                transactions_df, _ = normalize_df(transactions_df, normalize_by_row, standardize)
            else:
                transactions_df = normalize_df_with_scaler_params(transactions_df, crypto_name, preprocessing_config,
                                                                  scaler_dir)

    return transactions_df


def preprocess_data_point(df, crypto_name, preprocessing_config, scaler_dir=None):
    """
    Preprocess a data point containing cryptocurrency information for a single day and have it ready for prediction.
    The preprocessing, if normalization/standardization by column set, normalizes/standardizes the data using a
    previously fitted scaler (from the training/testing stages).
    :param df: The DataFrame containing the unprocessed cryptocurrency data (single row)
    :type df: DataFrame
    :param crypto_name
    :type crypto_name: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param scaler_dir: (Default None) The (overridden) directory where the scaler should be loaded. If None, the default
    `PREPROCESSED_DATA_DIR` is used.
    :type scaler_dir: str
    :return: The preprocessed DataFrame containing daily transaction in the last column.
    :rtype: DataFrame
    """

    price_column = preprocessing_config.price_column
    price_list = get_price_list(df, price_column)

    # Check if dates are outside price_list time range
    preprocessing_config.adjust_dates(df.index)

    # Preprocessing parameters
    window_size = preprocessing_config.window_size
    normalize = preprocessing_config.normalize
    normalize_by_row = preprocessing_config.normalize_by_row
    standardize = preprocessing_config.standardize

    transactions = [{'transaction': 'UNKNOWN', 'prices': price_list[-window_size:]}]  # Ensure window size
    transactions_df = build_transactions_df(transactions)

    # Data normalization
    if normalize:
        if normalize_by_row:
            transactions_df, _ = normalize_df(transactions_df, normalize_by_row, standardize)
        else:
            transactions_df = normalize_df_with_scaler_params(transactions_df, crypto_name, preprocessing_config,
                                                              scaler_dir)

    return transactions_df


def run_preprocessing_pipeline(crypto_name, preprocessing_config, save_data, save_roi):
    """
    Run the data preprocessing pipeline. The function returns a DataFrame containing data ready for training.
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.)
    :type crypto_name: str
    :param preprocessing_config: The preprocessing configuration object
    :type preprocessing_config: PreprocessingConfig
    :param save_data: Whether or not to save preprocessed data to a local file to avoid recalculation
    :type save_data: bool
    :param save_roi: Whether or not to save the preprocessing ROI to a local file for analysis
    :type save_roi: bool
    :return: A DataFrame ready for classification, consisting of a set of attributes and a class label.
    :rtype: DataFrame
    """

    logger.info("Running preprocessing pipeline...")
    logger.info("Preprocessing config:\n" + str(preprocessing_config))

    # Get DF
    historical_file = os.path.join(MASTER_DATA_DIR, "historical_{}.csv".format(CRYPTOCURRENCIES[crypto_name]))
    df = get_historical_df(historical_file)

    # Check if dates are outside price_list time range
    preprocessing_config.adjust_dates(df.index)

    # Load preprocessed data file if it exists
    if save_data:
        transactions_df = load_preprocessed_data(crypto_name, preprocessing_config)
        if transactions_df is not None:
            return transactions_df

    # Preprocess the data
    transactions_df = preprocess_dataframe(df, crypto_name, preprocessing_config, save_roi)

    # Save data
    if save_data:
        save_preprocessed_data(transactions_df, crypto_name, preprocessing_config)

    logger.info("Preprocessing pipeline complete!\n")

    return transactions_df


def run_preprocessing(cryptonalysis_config_grid, runs=1):
    """
    Run preprocessing using grid search hyperparameter optimization.
    :param cryptonalysis_config_grid
    :type cryptonalysis_config_grid: CryptonalysisConfigGrid
    :param runs: The number of times to run each preprocessing pipeline. Useful when using buy/sell probabilities to
    calculate average ROI.
    :type runs: int
    """
    logger.info("Config grid size: {}".format(cryptonalysis_config_grid.grid_size))

    preprocessed_df = None
    count = 1
    for i in range(runs):
        # Grid search preprocessing pipeline parameters
        for crypto in cryptonalysis_config_grid.cryptos:
            logger.info("Crypto: {}".format(crypto))
            for preprocessing_config in cryptonalysis_config_grid.preprocessing_config_grid:
                logger.info("Processing configuration {}/{}...".format(count, cryptonalysis_config_grid.grid_size))
                preprocessed_df = run_preprocessing_pipeline(crypto, preprocessing_config,
                                                             cryptonalysis_config_grid.save_preprocessing_data,
                                                             cryptonalysis_config_grid.save_preprocessing_roi)
                count += 1

    logger.debug(preprocessed_df.head())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Config file not given in args!')

    config_file = sys.argv[1]

    RUNS = 10  # Number of runs for ROI stats
    config_path = 'config'
    config_grid = load_cryptonalysis_config_grid(config_path, config_file)

    run_preprocessing(config_grid, RUNS)
