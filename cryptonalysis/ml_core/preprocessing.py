import logging
import pandas as pd
from pandas import DataFrame
from datetime import date
from transaction_builders import BiffPredictor
from market import market
from sklearn.preprocessing import StandardScaler


# Logging
LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger()

HISTORICAL_DATA_FILE = '../data/ethereum_historical.csv'


def get_historical_df(historical_file):
    """
    Gets a new DataFrame containing the historical data for a particular cryptocurrency.
    :param historical_file: The file path containing the historical data
    :return: DataFrame instance
    """
    logger.info("Loading historical data...")

    df = pd.read_csv(historical_file, sep='\t', thousands=',')

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

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


def run_data_pipeline(historical_file, starting_date=None, ending_date=None, price_column='Close', window_size=30,
                      crypto_name='ETH', normalize=True, normalize_by_row=False):
    """
    Run the data preprocessing pipeline. The function returns a DataFrame containing data ready for the classification
    task.
    :param historical_file: The file path containing the historical data
    :param starting_date: The date from which the data pipeline will begin (inclusive) (default is 01/01/2017)
    :param ending_date: The date in which the data pipeline will end (non-inclusive) (default is today)
    :param price_column: The name of the column containing the crypto price (default is 'Close')
    :param window_size: The size of the price window (in days) to use in the transaction prediction (default is 30)
    :param crypto_name: The cryptocurrency name (e.g., ETH, BTC, etc.) (default is 'ETH')
    :param normalize: Whether or not the data should be normalized (default is True)
    :param normalize_by_row: Whether or not the data should be normalize by row or column (ignored if normalize=False)
    (default is False)
    :return: A DataFrame ready for classification, consisting of a set of attributes and a class label.
    """

    logger.info("Running preprocessing pipeline...")

    starting_date = starting_date or date(2017, 1, 1)  # First date for ETH is 2015 8 7
    ending_date = ending_date or date.today()

    logger.info("Parameters:\nHistorical file: '{0}', Starting date: {1}, Ending date: {2}, Price column: '{3}', "
                "Window size: {4} days, Crypto name: '{5}', Normalize: {6}, Normalize by row: {7}".format(
                    historical_file, starting_date, ending_date, price_column, window_size, crypto_name, normalize,
                    normalize_by_row))

    # 1. Get DF
    df = get_historical_df(historical_file)

    # 2. Get aggregated DFs
    daily_df, weekly_df, monthly_df = get_aggregated_dfs(df)

    # 3. Get transaction data
    price_list = daily_df[price_column]
    opening_price_list = daily_df['Open']

    biff = BiffPredictor(market, starting_date, price_list, opening_price_list, window_size, crypto_name, ending_date)
    biff.run_predictor()

    # 4. Get transactions DataFrame
    transactions_df = build_transactions_df(biff.transactions)

    logger.info('Buy: {0}'.format(len(transactions_df[transactions_df['transaction'] == 1])))
    logger.info('Sell: {0}'.format(len(transactions_df[transactions_df['transaction'] == 0])))

    # 5. Data normalization
    if normalize:
        transactions_df = normalize_df(transactions_df, normalize_by_row)

    return transactions_df


if __name__ == '__main__':
    preprocessed_df = run_data_pipeline(HISTORICAL_DATA_FILE)
    logger.info(preprocessed_df.head())
