import pandas as pd
from pandas import DataFrame
from datetime import date
from transaction_builders import BiffPredictor, ProbabilityPredictor, RandomPredictor
from market import market
import logging

# Logging
LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger()

HISTORICAL_DATA_FILE = '../jupyter/ethereum_historical.csv'


def get_historical_df():
    """
    Gets a new DataFrame containing the historical data for a particular cryptocurrency.
    :return: DataFrame instance
    """
    df = pd.read_csv(HISTORICAL_DATA_FILE, sep='\t', thousands=',')

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


def run_pipeline():
    # 1. Get DF
    df = get_historical_df()

    # 2. Get aggregated DFs
    daily_df, weekly_df, monthly_df = get_aggregated_dfs(df)

    # 3. Get transaction data
    # Required predictor parameters
    starting_date = date(2017, 1, 1)  # 2015 8 7
    column = 'Close'
    price_list = daily_df[column][starting_date:]
    opening_price_list = daily_df['Open'][starting_date:]
    window_size = 30
    crypto_name = 'ETH'

    biff = BiffPredictor(market, starting_date, price_list, opening_price_list, window_size, crypto_name)
    biff.run_predictor()

    lucky = ProbabilityPredictor(market, starting_date, price_list, opening_price_list, window_size, crypto_name,
                                 prob_buy=0.63, prob_sell=0.49)
    lucky.run_predictor()

    random = RandomPredictor(market, starting_date, price_list, opening_price_list, window_size, crypto_name)
    random.run_predictor()

    # 4. Get transactions DataFrame
    transactions_df = build_transactions_df(biff.transactions)

    logger.info('Buy: {0}'.format(len(transactions_df[transactions_df['transaction'] == 1])))
    logger.info('Sell: {0}'.format(len(transactions_df[transactions_df['transaction'] == 0])))


if __name__ == '__main__':
    run_pipeline()
