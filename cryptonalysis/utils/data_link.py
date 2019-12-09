import logging
import pandas as pd
from cryptonalysis.ml_core.preprocessing import CRYPTOCURRENCIES
from datetime import date, timedelta
from misc_utils import parse_date


# Logging
logger = logging.getLogger()

API_URLS = {
    crypto_short: "https://coinmarketcap.com/currencies/{}/historical-data/".format(crypto_long)
    for crypto_short, crypto_long in CRYPTOCURRENCIES.items()
}


def fetch_crypto_data(crypto_name, start_date, end_date):
    """
    Fetch cryptocurrency data from a remote API between a given start date and a given end date.
    :param crypto_name
    :type crypto_name: str
    :param start_date: Inclusive starting date
    :type start_date: date
    :param end_date: Inclusive ending date.
    :type end_date: date
    :return: A cleansed DataFrame with cryptocurrency data.
    :rtype: pd.DataFrame
    """
    current_date = parse_date('today')
    if start_date > current_date or end_date > current_date:
        raise ValueError('start_date and end_date must be before or equal to today!')

    if start_date >= end_date:
        raise ValueError('start_date must be before end_date!')

    start_date_str = start_date.strftime('%Y%m%d')
    adjusted_end_date = end_date - timedelta(days=1)  # The API always returns an extra day
    end_date_str = adjusted_end_date.strftime('%Y%m%d')
    crypto_endpoint = "{}?start={}&end={}".format(API_URLS[crypto_name], start_date_str, end_date_str)

    df = pd.read_html(crypto_endpoint)[2]  # Format change

    # Clean up column names
    def replace(s):
        return s.replace('*', '')
    df = df.rename(replace, axis='columns')

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


def get_crypto_data_for_date(crypto_name, for_date, window_size):
    """
    Get the cryptocurrency data for a given date (minus 1 day) with a given time window size in days.
    Given that the data for the given date is probably not yet ready, the last data point will actually correspond to
    one day before the `for_date`.
    :param crypto_name
    :param for_date: The date for which the data will be retrieved (in practice it is for the previous day).
    :type for_date: date
    :param window_size: The number of days before the given date worth of data that will be retrieved.
    For example, if the value is 10, 10 rows of data will be retrieved between the day before `for_date` and -10 days,
    both inclusive.
    :type window_size: int
    :return: A DataFrame with cryptocurrency data for a single given date.
    :rtype: pd.DataFrame
    """
    current_date = parse_date('today')
    if for_date > current_date:
        raise ValueError('for_date cannot be after today!')

    logger.info("Getting {} data for {} with window size {}...".format(crypto_name, for_date, window_size))

    start_date = for_date - timedelta(days=window_size)
    end_date = for_date - timedelta(days=1)
    crypto_data = fetch_crypto_data(crypto_name, start_date, end_date)

    return crypto_data


if __name__ == '__main__':
    crypto = 'ETH'
    today = parse_date('today')
    data = get_crypto_data_for_date(crypto, today, 50)
    print data
