import pandas as pd
from pandas import DataFrame
from transaction_builders import *
from market import market
from sklearn.preprocessing import StandardScaler
from config import load_config, PreprocessingConfig
import os


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


def build_transactions_df(transactions):
    """
    Build a DataFrame ready for classification (i.e., X and y) based on a dictionary of transactions.
    :param transactions: A list of dictionaries of the form
    [{'transaction': 'TRANSACTION_TYPE', 'prices': [price1, price2, ...]}, ...]
    :return: A DataFrame with N price columns and a transaction column (1=BUY, 0=SELL)
    """

    logger.info("Getting transactions DataFrame...")

    prices = list(transactions[0]['prices'])

    # Price columns
    df_columns = ["price {0}".format(column_name) for column_name in range(1, len(prices) + 1)]
    df_columns.append('transaction')

    df = DataFrame(columns=df_columns)
    for row in range(len(transactions)):
        prices = list(transactions[row]['prices'])
        transaction = 1 if transactions[row]['transaction'] == 'BUY' else 0
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
    An example value returned would be '../data/pre_ETH_ProbabilityPredictor_2018-01-01-2018-07-10_win30_norm_col'.
    An example filename would be 'pre_ProbabilityPredictor_2018-01-01-2018-07-10_win30_norm_col.csv', meaning the data
    was obtained using the ProbabilityPredictor, using historical data from 2018-01-01 to 2018-07-10, a window size of
    30, and normalization by column.
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
    """
    norm_string = '_norm_{0}'.format('row' if normalize_by_row else 'col') if normalize else ''
    filename = 'pre_{0}_{1}({2})_{3}-{4}_win{5}{6}_b{7}s{8}.csv'.format(crypto_name, predictor_class.__name__,
                                                                        lookahead_days, starting_date, ending_date,
                                                                        window_size, norm_string, prob_buy, prob_sell)
    file_path = os.path.join(MASTER_DATA_DIR, filename)
    return file_path


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
    :return:
    """
    preprocessed_data_filename = get_data_filename(crypto_name, predictor_class, lookahead_days, starting_date,
                                                   ending_date, window_size, normalize, normalize_by_row, prob_buy,
                                                   prob_sell)
    if os.path.isfile(preprocessed_data_filename):
        logger.info("Preprocessed datafile '{0}'' already exists. "
                    "Loading file and skipping preprocessing pipeline...".format(preprocessed_data_filename))
        transactions_df = pd.read_csv(preprocessed_data_filename)
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

    data_file_path = get_data_filename(crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                                       window_size, normalize, normalize_by_row, prob_buy, prob_sell)
    logger.info("Saving data to file '{0}'".format(data_file_path))
    transactions_df.to_csv(data_file_path, index=False)


def run_data_pipeline(crypto_name, preprocessing_config, **kwargs):
    """
    Run the data preprocessing pipeline. The function returns a DataFrame containing data ready for the classification
    task.
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.)
    :type crypto_name: str
    :param preprocessing_config: The preprocessing configuration object
    :type preprocessing_config: PreprocessingConfig
    :param kwargs: Dictionary of parameters used by the predictor_class (e.g. ProbabilityPredictor's  'prob_buy' and
    'prob_sell' parameters).
    :return: A DataFrame ready for classification, consisting of a set of attributes and a class label.
    """

    logger.info("Running preprocessing pipeline...")

    historical_file = os.path.join(MASTER_DATA_DIR, "historical_{}.csv".format(CRYPTOCURRENCIES[crypto_name]))

    # Get runtime parameters
    starting_date = preprocessing_config.start_date  # First date for ETH is 2015 8 7
    ending_date = preprocessing_config.end_date

    # Predictor parameters
    lookahead_days = kwargs['lookahead_days'] if 'lookahead_days' in kwargs else CryptoPredictor.LOOKAHEAD_DAYS
    starting_investment = kwargs['starting_investment'] if 'starting_investment' in kwargs else None
    daily_allowance = kwargs['daily_allowance'] if 'daily_allowance' in kwargs else None
    prob_buy = kwargs['prob_buy'] if 'prob_buy' in kwargs else CryptoPredictor.PROB_BUY
    prob_sell = kwargs['prob_sell'] if 'prob_sell' in kwargs else CryptoPredictor.PROB_SELL

    # 0. Get DF
    df = get_historical_df(historical_file)

    logger.info("Parameters:\nHistorical file: '{0}', Start: {1}, End: {2}, Price col.: '{3}', "
                "Window size: {4} days, Crypto: '{5}', Norm.: {6}, Norm. by row: {7}".format(
                    historical_file, starting_date, ending_date, preprocessing_config.price_column,
                    preprocessing_config.window_size, crypto_name, preprocessing_config.normalize,
                    preprocessing_config.normalize_by_row))

    # 1. Load preprocessed data file if it exists
    transactions_df = load_data_file(crypto_name, preprocessing_config.predictor_cls, lookahead_days, starting_date,
                                     ending_date, preprocessing_config.window_size, preprocessing_config.normalize,
                                     preprocessing_config.normalize_by_row, prob_buy, prob_sell)
    if transactions_df is not None:
        return transactions_df

    # 2. Get aggregated DFs
    daily_df, weekly_df, monthly_df = get_aggregated_dfs(df)

    # 3. Get transaction data
    price_list = daily_df[preprocessing_config.price_column]

    # 4. Run transaction builder
    predictor = \
        preprocessing_config.predictor_cls(market, starting_date, price_list, preprocessing_config.window_size,
                                           crypto_name, ending_date, starting_investment=starting_investment,
                                           daily_allowance=daily_allowance, lookahead_days=lookahead_days,
                                           prob_buy=prob_buy, prob_sell=prob_sell)
    predictor.run_predictor(preprocessing_config.save_roi)

    # 5. Get transactions DataFrame
    transactions_df = build_transactions_df(predictor.transactions)

    logger.info('Buy: {0}'.format(len(transactions_df[transactions_df['transaction'] == 1])))
    logger.info('Sell: {0}'.format(len(transactions_df[transactions_df['transaction'] == 0])))

    # 6. Data normalization
    if preprocessing_config.normalize:
        transactions_df = normalize_df(transactions_df, preprocessing_config.normalize_by_row)

    # 7. Save data
    if preprocessing_config.save_data:
        save_data_file(transactions_df, crypto_name, preprocessing_config.predictor_cls, lookahead_days, starting_date,
                       ending_date, preprocessing_config.window_size, preprocessing_config.normalize,
                       preprocessing_config.normalize_by_row, prob_buy, prob_sell)

    logger.info("Preprocessing pipeline complete!\n")

    return transactions_df


if __name__ == '__main__':
    RUNS = 5  # Number of runs for ROI stats
    config_path = os.path.join(os.path.pardir, os.path.join(os.path.pardir, 'config'))
    cryptonalysis_config = load_config(config_path)

    preprocessed_df = None
    for i in range(RUNS):
        preprocessed_df = run_data_pipeline(cryptonalysis_config.crypto, cryptonalysis_config.preprocessing_config,
                                            **cryptonalysis_config.preprocessing_config.predictor_params)
    logger.debug(preprocessed_df.head())
