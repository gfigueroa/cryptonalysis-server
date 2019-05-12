import pandas as pd
from pandas import DataFrame
from transaction_builders import *
from market import market
from sklearn.preprocessing import StandardScaler
import config
import os


# Logging
logging.basicConfig(level=config.LOGGING_LEVEL)
logger = logging.getLogger()

CRYPTO_CURRENCIES = {
    'ETH': "ethereum",
    'BTC': "bitcoin",
    'XRP': "ripple",
    'LTC': "litecoin",
    'USDT': "tether"
}
DATA_FOLDER = os.path.join(os.path.pardir, 'data')
HISTORICAL_DATA_FILE = os.path.join(DATA_FOLDER, 'ethereum_historical.csv')
HISTORICAL_DATA_FILES = {
    key: os.path.join(DATA_FOLDER, "{}_historical.csv".format(value))
    for (key, value) in CRYPTO_CURRENCIES.iteritems()
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
    # type: (pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    Get time aggregated (daily, weekly, and monthly) DataFrames from historical DF.
    :param historical_df: The historical DF to aggregate
    :return: a tuple of the form (DataFrame, DataFrame, DataFrame), containing a daily DF, weekly DF, and monthly DF,
    respectively.
    """

    logger.info("Aggregating data...")

    df = historical_df.copy(deep=True)
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
                      normalize_by_row):
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
    :type predictor_class: ``classobj``
    :param lookahead_days: The number of lookahead days used by the predictor_class
    :param starting_date: The date from which the data pipeline began (inclusive)
    :param ending_date: The date in which the data pipeline ended
    :param window_size: The size of the price window (in days) used in the transaction prediction
    :param normalize: Whether or not the data was normalized
    :param normalize_by_row: Whether or not the data was normalized by row or column (ignored if normalize=False)
    (default is False)
    """
    norm_string = '_norm_{0}'.format('row' if normalize_by_row else 'col') if normalize else ''
    filename = 'pre_{0}_{1}({2})_{3}-{4}_win{5}{6}.csv'.format(crypto_name, predictor_class.__name__, lookahead_days,
                                                               starting_date, ending_date, window_size, norm_string)
    file_path = os.path.join(DATA_FOLDER, filename)
    return file_path


def load_data_file(crypto_name, predictor_class, lookahead_days, starting_date, ending_date, window_size, normalize,
                   normalize_by_row):
    """
    Load the DataFrame (if saved) as a CSV containing data ready for the classification task.
    :param crypto_name
    :type crypto_name: str
    :param predictor_class: The class used to build transactions (default is BiffPredictor)
    :type predictor_class: ``classobj``
    :param lookahead_days: The number of lookahead days used by the predictor_class
    :param starting_date: The date from which the data pipeline began (inclusive)
    :param ending_date: The date in which the data pipeline ended
    :param window_size: The size of the price window (in days) used in the transaction prediction
    :param normalize: Whether or not the data was normalized
    :param normalize_by_row: Whether or not the data was normalized by row or column (ignored if normalize=False)
    :return:
    """
    preprocessed_data_filename = get_data_filename(crypto_name, predictor_class, lookahead_days, starting_date,
                                                   ending_date, window_size, normalize, normalize_by_row)
    if os.path.isfile(preprocessed_data_filename):
        logger.info("Preprocessed datafile '{0}'' already exists. "
                    "Loading file and skipping preprocessing pipeline...".format(preprocessed_data_filename))
        transactions_df = pd.read_csv(preprocessed_data_filename)
        return transactions_df
    else:
        return None


def save_data_file(transactions_df, crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                   window_size, normalize, normalize_by_row):
    """
    Save the DataFrame containing data ready for the classification task as a CSV file.
    :param crypto_name
    :type crypto_name: str
    :param transactions_df: The DataFrame containing the actual preprocessed data
    :param predictor_class: The class used to build transactions (default is BiffPredictor)
    :type predictor_class: ``classobj``
    :param lookahead_days: The number of lookahead days used by the predictor_class
    :param starting_date: The date from which the data pipeline began (inclusive)
    :param ending_date: The date in which the data pipeline ended
    :param window_size: The size of the price window (in days) used in the transaction prediction
    :param normalize: Whether or not the data was normalized
    :param normalize_by_row: Whether or not the data was normalized by row or column (ignored if normalize=False)
    (default is False)
    """

    data_file_path = get_data_filename(crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                                       window_size, normalize, normalize_by_row)
    logger.info("Saving data to file '{0}'".format(data_file_path))
    transactions_df.to_csv(data_file_path, index=False)


def run_data_pipeline(historical_file, starting_date=None, ending_date=None, price_column='Close',
                      predictor_class=BiffPredictor, window_size=30, crypto_name='ETH', normalize=True,
                      normalize_by_row=False, save_data=True, **kwargs):
    """
    Run the data preprocessing pipeline. The function returns a DataFrame containing data ready for the classification
    task.
    :param historical_file: The file path containing the historical data
    :param starting_date: The date from which the data pipeline will begin (inclusive) (default is 01/01/2017)
    :param ending_date: The date in which the data pipeline will end (non-inclusive) (default is today)
    :param price_column: The name of the column containing the crypto price (default is 'Close')
    :param predictor_class: The CryptoPredictor subclass used to build transactions (default is BiffPredictor)
    :type predictor_class: ``classobj``
    :param window_size: The size of the price window (in days) to use in the transaction prediction (default is 30)
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.) (default is 'ETH')
    :type crypto_name: str
    :param normalize: Whether or not the data should be normalized (default is True)
    :param normalize_by_row: Whether or not the data should be normalized by row or column (ignored if normalize=False)
    (default is False)
    :param save_data: Whether or not to save the DataFrame containing the data for classification as a csv file
    :param kwargs: Dictionary of parameters used by the predictor_class (e.g. ProbabilityPredictor's  'prob_buy' and
    'prob_sell' parameters).
    :return: A DataFrame ready for classification, consisting of a set of attributes and a class label.
    """

    logger.info("Running preprocessing pipeline...")

    # Get runtime parameters
    starting_date = starting_date or date(2017, 1, 1)  # First date for ETH is 2015 8 7

    lookahead_days = kwargs['lookahead_days'] if 'lookahead_days' in kwargs else CryptoPredictor.LOOKAHEAD_DAYS
    starting_investment = kwargs['starting_investment'] if 'starting_investment' in kwargs else None
    daily_allowance = kwargs['daily_allowance'] if 'daily_allowance' in kwargs else None
    prob_buy = kwargs['prob_buy'] if 'prob_buy' in kwargs else 1
    prob_sell = kwargs['prob_sell'] if 'prob_sell' in kwargs else 1

    # 0. Get DF
    df = get_historical_df(historical_file)
    if not ending_date:
        ending_date = df['Date'].dt.date.iat[-1]

    logger.info("Parameters:\nHistorical file: '{0}', Start: {1}, End: {2}, Price col.: '{3}', "
                "Window size: {4} days, Crypto: '{5}', Norm.: {6}, Norm. by row: {7}".format(
                    historical_file, starting_date, ending_date, price_column, window_size, crypto_name, normalize,
                    normalize_by_row))

    # 1. Load preprocessed data file if it exists
    transactions_df = load_data_file(crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                                     window_size, normalize, normalize_by_row)
    if transactions_df is not None:
        return transactions_df

    # 2. Get aggregated DFs
    daily_df, weekly_df, monthly_df = get_aggregated_dfs(df)

    # 3. Get transaction data
    price_list = daily_df[price_column]

    # 4. Run transaction builder
    predictor = predictor_class(market, starting_date, price_list, window_size, crypto_name, ending_date,
                                starting_investment=starting_investment, daily_allowance=daily_allowance,
                                lookahead_days=lookahead_days, prob_buy=prob_buy, prob_sell=prob_sell)
    predictor.run_predictor()

    # 5. Get transactions DataFrame
    transactions_df = build_transactions_df(predictor.transactions)

    logger.info('Buy: {0}'.format(len(transactions_df[transactions_df['transaction'] == 1])))
    logger.info('Sell: {0}'.format(len(transactions_df[transactions_df['transaction'] == 0])))

    # 6. Data normalization
    if normalize:
        transactions_df = normalize_df(transactions_df, normalize_by_row)

    # 7. Save data
    if save_data:
        save_data_file(transactions_df, crypto_name, predictor_class, lookahead_days, starting_date, ending_date,
                       window_size, normalize, normalize_by_row)

    logger.info("Preprocessing pipeline complete!\n")

    return transactions_df


if __name__ == '__main__':
    start_date = date(2018, 1, 1)
    predictor_cls = BiffPredictorSmart
    predictor_params = {
        'prob_buy': 0.8,
        'prob_sell': 0.8,
        'starting_investment': 100,
        'daily_allowance': 5,
        'lookahead_days': 4
    }
    save = False
    preprocessed_df = None
    for i in range(5):
        preprocessed_df = run_data_pipeline(HISTORICAL_DATA_FILE, starting_date=start_date,
                                            predictor_class=predictor_cls, save_data=save, **predictor_params)
    logger.debug(preprocessed_df.head())
