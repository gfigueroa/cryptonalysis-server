import logging
import os
import pandas as pd
import sys
from cryptonalysis.config import load_cryptonalysis_config_grid, PreprocessingConfig, CryptonalysisConfigGrid
from datetime import date
from market import market
from pandas import DataFrame, to_datetime
from sklearn.preprocessing import StandardScaler
from transaction_builders import TRANSACTION_TYPE, BiffPredictor

# Logging
logger = logging.getLogger()

# Constants
MASTER_DATA_DIR = os.path.join(os.path.pardir, 'master_data')
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

    return df


def get_aggregated_dfs(historical_df):
    """
    Get time aggregated (daily, weekly, and monthly) DataFrames from historical DF.
    :param historical_df: The historical DF to aggregate
    :type historical_df: DataFrame
    :return: a tuple of the form (DataFrame, DataFrame, DataFrame), containing a daily DF, weekly DF, and monthly DF,
    respectively.
    :rtype: (DataFrame, DataFrame, DataFrame)
    """

    logger.info("Aggregating data...")

    df = historical_df.copy(deep=True)  # type: DataFrame
    df.insert(1, 'Week', pd.PeriodIndex(df.Date, freq='W'))
    df.insert(2, 'Month', pd.PeriodIndex(df.Date, freq='M'))

    daily_df = df.set_index('Date').loc[:, 'Open':]
    weekly_df = df.groupby(by=['Week']).mean()
    monthly_df = df.groupby(by=['Month']).mean()

    return daily_df, weekly_df, monthly_df


def get_price_list(df, price_column):
    """
    Get a DataFrame with a single column of prices given a composite DataFrame and the name of the price column to
    consider.
    :param df: The original cryptocurrency DataFrame.
    :type df: DataFrame
    :param price_column: The name of the column containing the crypto price to consider.
    :type price_column: str
    :return: A DataFrame containing a single column with cryptocurrency prices
    :rtype: DataFrame
    """

    # Get aggregated DFs
    daily_df, weekly_df, monthly_df = get_aggregated_dfs(df)

    # Get transaction data
    price_list = daily_df[price_column]

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

    prices = list(transactions[0]['prices'])

    # Price columns
    df_columns = ["price {0}".format(column_name) for column_name in range(1, len(prices) + 1)]
    df_columns.append('transaction')

    df = DataFrame(columns=df_columns)
    for row in range(len(transactions)):
        prices = list(transactions[row]['prices'])
        transaction = TRANSACTION_TYPE[transactions[row]['transaction']]
        df.loc[row] = prices + [transaction]

    df['transaction'] = df['transaction'].astype(int)

    return df


def normalize_df(df, by_row):
    """
    Normalize a DataFrame either by row using feature scaling (by_row=True), or using scikit-learn's by-column
    StandardScaler (by_row=False).
    A value is normalized in the row using the feature scaling formula to be in the range [0, 1]:
    x' = (x - x_min) / (x_max - x_min)
    :param df: The DataFrame to normalize
    :param by_row: If True, normalizes by row using feature scaling, if False, uses scikit-learn's FeatureScaler, which
    scales by column and using variance.
    :return: A new DataFrame with each row normalized
    :rtype: DataFrame
    """

    logger.info("Normalizing data...")

    if by_row:
        norm_df = df.copy(deep=True)
        for row in range(len(df)):
            norm_df.iloc[row, :-1] = (df.iloc[row, :-1] - df.iloc[row, :-1].min()) / \
                                     (df.iloc[row, :-1].max() - df.iloc[row, :-1].min())
    else:
        scaler = StandardScaler()
        scaler.fit(df.iloc[:, :-1])
        norm_df = DataFrame(scaler.transform(df.iloc[:, :-1]))
        norm_df['transaction'] = df['transaction']

    return norm_df


def get_data_filename(crypto_name, predictor_class, lookahead_days, starting_date, ending_date, window_size, normalize,
                      normalize_by_row, prob_buy, prob_sell):
    """
    Get the path and filename used for the data file containing the preprocessed data based on the preprocessing
    parameters.

    >>> get_data_filename('ETH', BiffPredictor, 1, date(2018, 1, 1), date(2018, 7, 10), 30, True, False, 1, 1)
    'pre_ETH_BiffPredictor(1)_2018-01-01-2018-07-10_win30_norm_col_b1s1.csv'
    >>> get_data_filename('ETH', BiffPredictor, 1, date(2018, 1, 1), date(2018, 7, 10), 30, False, False, 1, 1)
    'pre_ETH_BiffPredictor(1)_2018-01-01-2018-07-10_win30_b1s1.csv'
    >>> get_data_filename('ETH', BiffPredictor, 1, date(2018, 1, 1), date(2018, 7, 10), 30, False, True, 1, 1)
    'pre_ETH_BiffPredictor(1)_2018-01-01-2018-07-10_win30_b1s1.csv'

    :param crypto_name
    :type crypto_name: str
    :param predictor_class: The class used to build transactions (default is BiffPredictor)
    :type predictor_class: type
    :param lookahead_days: The number of lookahead days used by the predictor_class
    :param starting_date: The date from which the data pipeline began (inclusive)
    :param ending_date: The date in which the data pipeline ended
    :param window_size: The size of the price window (in days) used in the transaction prediction
    :param normalize: Whether or not the data was normalized
    :param normalize_by_row: Whether or not the data was normalized by row or column (ignored if normalize=False)
    (default is False)
    :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
    when it actually has to buy.
    :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
    when it actually has to sell.
    :return A string with the name of a preprocessed data file given a series of config parameters
    :rtype: str
    """
    norm_string = '_norm_{0}'.format('row' if normalize_by_row else 'col') if normalize else ''
    filename = 'pre_{0}_{1}({2})_{3}-{4}_win{5}{6}_b{7}s{8}.csv'.format(crypto_name, predictor_class.__name__,
                                                                        lookahead_days, starting_date, ending_date,
                                                                        window_size, norm_string, prob_buy, prob_sell)
    return filename


def load_data_file(crypto_name, predictor_class, lookahead_days, starting_date, ending_date, window_size, normalize,
                   normalize_by_row, prob_buy, prob_sell):
    """
    Load the DataFrame (if saved) as a CSV containing data ready for the classification task.
    :param crypto_name
    :type crypto_name: str
    :param predictor_class: The class used to build transactions (default is BiffPredictor)
    :type predictor_class: type
    :param lookahead_days: The number of lookahead days used by the predictor_class
    :param starting_date: The date from which the data pipeline began (inclusive)
    :param ending_date: The date in which the data pipeline ended
    :param window_size: The size of the price window (in days) used in the transaction prediction
    :param normalize: Whether or not the data was normalized
    :param normalize_by_row: Whether or not the data was normalized by row or column (ignored if normalize=False)
    :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
    when it actually has to buy.
    :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
    when it actually has to sell.
    :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
    when it actually has to buy.
    :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
    when it actually has to sell.
    :return: a DataFrame
    :rtype: DataFrame
    """
    data_filename = get_data_filename(crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                                      window_size, normalize, normalize_by_row, prob_buy, prob_sell)
    data_file_path = os.path.join(MASTER_DATA_DIR, data_filename)
    if os.path.isfile(data_file_path):
        logger.info("Preprocessed datafile '{0}'' already exists. "
                    "Loading file and skipping preprocessing pipeline...".format(data_file_path))
        transactions_df = pd.read_csv(data_file_path)
        return transactions_df
    else:
        return None


def save_data_file(transactions_df, crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                   window_size, normalize, normalize_by_row, prob_buy, prob_sell):
    """
    Save the DataFrame containing data ready for the classification task as a CSV file.
    :param crypto_name
    :type crypto_name: str
    :param transactions_df: The DataFrame containing the actual preprocessed data
    :param predictor_class: The class used to build transactions (default is BiffPredictor)
    :type predictor_class: type
    :param lookahead_days: The number of lookahead days used by the predictor_class
    :param starting_date: The date from which the data pipeline began (inclusive)
    :param ending_date: The date in which the data pipeline ended
    :param window_size: The size of the price window (in days) used in the transaction prediction
    :param normalize: Whether or not the data was normalized
    :param normalize_by_row: Whether or not the data was normalized by row or column (ignored if normalize=False)
    (default is False)
    :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
    when it actually has to buy.
    :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
    when it actually has to sell.
    """

    data_filename = get_data_filename(crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                                      window_size, normalize, normalize_by_row, prob_buy, prob_sell)
    data_file_path = os.path.join(MASTER_DATA_DIR, data_filename)
    logger.info("Saving data to file '{0}'".format(data_file_path))
    transactions_df.to_csv(data_file_path, index=False)


def preprocess_dataframe(df, crypto_name, predictor_cls, price_column, window_size, normalize, normalize_by_row,
                         starting_date=None, ending_date=None, starting_investment=None, daily_allowance=None,
                         lookahead_days=None, prob_buy=None, prob_sell=None, save_roi=None, run_predictor=True):
    """
    Preprocess a dataframe containing cryptocurrency information and have it ready for classification.
    The preprocessing, by default, runs a `CryptoPredictor`, which is necessary for training and testing models.
    This step can be skipped when only daily prediction is required.
    :param df: The DataFrame containing the unprocessed cryptocurrency data
    :type df: DataFrame
    :param crypto_name
    :type crypto_name: str
    :param predictor_cls: The class used to build transactions (default is BiffPredictor)
    :type predictor_cls: type
    :param price_column: The name of the column containing the price to use for training.
    :type price_column: str
    :param window_size: The size of the price window (in days) used in the transaction prediction
    :type window_size: int
    :param normalize: Whether or not the data was normalized
    :type normalize: bool
    :param normalize_by_row: Whether or not the data was normalized by row or column (ignored if normalize=False)
    (default is False)
    :type normalize_by_row: bool
    :param starting_date: The date from which the data pipeline began (inclusive)
    :param ending_date: The date in which the data pipeline ended
    :param starting_investment
    :param daily_allowance
    :param lookahead_days: The number of lookahead days used by the predictor_class
    :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
    when it actually has to buy.
    :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
    when it actually has to sell.
    :param save_roi: Whether or not to save the ROI of this predictor run in a file
    :type save_roi: bool
    :param run_predictor: Whether or not to run the `CryptoPredictor` specified in the `predictor_cls` parameter.
    :type run_predictor: bool
    :return: A preprocessing DataFrame containing daily transactions in the last column.
    :rtype: DataFrame
    """

    price_list = get_price_list(df, price_column)

    # Run transaction builder
    if run_predictor:
        predictor = predictor_cls(market, price_list, window_size, crypto_name, starting_date, ending_date,
                                  starting_investment=starting_investment, daily_allowance=daily_allowance,
                                  lookahead_days=lookahead_days, prob_buy=prob_buy, prob_sell=prob_sell)
        predictor.run_predictor(save_roi)

        # Get transactions DataFrame
        transactions_df = build_transactions_df(predictor.transactions)

        logger.info('Buy: {0}'.format(len(transactions_df[transactions_df['transaction'] == TRANSACTION_TYPE['BUY']])))
        logger.info('Sell: {0}'.format(len(transactions_df[transactions_df['transaction'] == TRANSACTION_TYPE['SELL']])))
    else:  # Pure prediction (unknown target)
        transactions = [{'transaction': 'UNKNOWN', 'prices': price_list[-window_size:]}]  # Ensure window size
        transactions_df = build_transactions_df(transactions)

    # Data normalization
    # TODO: Normalization should be part of training stage and column-based normalization should be stored with model
    if normalize:
        transactions_df = normalize_df(transactions_df, normalize_by_row)

    return transactions_df


def run_preprocessing_pipeline(crypto_name, preprocessing_config, save_data, save_roi):
    """
    Run the data preprocessing pipeline. The function returns a DataFrame containing data ready for the classification
    task.
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

    # Adjust end_date to latest date in historical df
    latest_date = to_datetime(df['Date'].iloc[-1]).date()
    preprocessing_config.adjust_end_date(latest_date)

    # Preprocessing parameters
    predictor_cls = preprocessing_config.predictor_cls
    starting_date = preprocessing_config.start_date
    ending_date = preprocessing_config.end_date
    price_column = preprocessing_config.price_column
    window_size = preprocessing_config.window_size
    normalize = preprocessing_config.normalize
    normalize_by_row = preprocessing_config.normalize_by_row

    # Predictor parameters
    lookahead_days = preprocessing_config.predictor_params['lookahead_days']
    starting_investment = preprocessing_config.predictor_params['starting_investment']
    daily_allowance = preprocessing_config.predictor_params['daily_allowance']
    prob_buy = preprocessing_config.predictor_params['prob_buy']
    prob_sell = preprocessing_config.predictor_params['prob_sell']

    # Load preprocessed data file if it exists
    if save_data:
        transactions_df = load_data_file(crypto_name, predictor_cls, lookahead_days, starting_date, ending_date,
                                         window_size, normalize, normalize_by_row, prob_buy, prob_sell)
        if transactions_df is not None:
            return transactions_df

    # Preprocess the data
    transactions_df = preprocess_dataframe(df, crypto_name, predictor_cls, price_column, window_size, normalize,
                                           normalize_by_row, starting_date, ending_date, starting_investment,
                                           daily_allowance, lookahead_days, prob_buy, prob_sell, save_roi)

    # Save data
    if save_data:
        save_data_file(transactions_df, crypto_name, predictor_cls, lookahead_days, starting_date, ending_date,
                       window_size, normalize, normalize_by_row, prob_buy, prob_sell)

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
    config_path = os.path.join(os.path.pardir, os.path.join(os.path.pardir, 'config'))
    config_grid = load_cryptonalysis_config_grid(config_path, config_file)

    run_preprocessing(config_grid, RUNS)
