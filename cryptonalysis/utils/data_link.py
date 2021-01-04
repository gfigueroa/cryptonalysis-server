import logging
import pandas as pd
import os
import time
from cryptonalysis.ml_core.preprocessing import CRYPTOCURRENCIES, MASTER_DATA_DIR, get_historical_df
from datetime import date, timedelta
from misc_utils import parse_date
from selenium import webdriver


# Logging
logger = logging.getLogger()
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

API_URLS = {
    crypto_short: "https://coinmarketcap.com/currencies/{}/historical-data/".format(crypto_long)
    for crypto_short, crypto_long in map(lambda (k, v): (k, v) if k != 'XRP' else (k, k), CRYPTOCURRENCIES.items())
}
CHROME_DRIVER_PATH = 'resources/chromedriver'


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

    crypto_endpoint = API_URLS[crypto_name]

    # Wait for dynamic content to load
    driver = webdriver.Chrome(CHROME_DRIVER_PATH)
    time.sleep(5)
    driver.get(crypto_endpoint.lower())

    df = pd.read_html(driver.page_source)[0]

    # Clean up column names
    def replace(s):
        return s.replace('*', '')
    df = df.rename(replace, axis='columns')

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Convert numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

    # Cleanup
    # Remove '-' from Market Cap
    df.loc[df['Market Cap'] == '-', 'Market Cap'] = 0
    # Remove extra chars and convert to numeric
    for col in numeric_columns:
        # Convert to numeric
        df[col] = df[col].str.replace(',', '')
        df[col] = df[col].str.replace('$', '')
        df[col] = pd.to_numeric(df[col])
        df = df.fillna(0)

    # Set index
    df = df.set_index('Date').loc[:, 'Open':]

    # Filter dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    df = df.loc[start_date_str:end_date_str]

    # CoinMarketCap only returns data up to a certain date
    try:
        df.loc[start_date_str]
    except KeyError:
        raise IOError("You're too late to fetch data, the start date is too old! ")

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


def update_new_data(crypto_name, end_date):
    """
    Automatically fetch and update the new_CRYPTO.csv files with new data up to the given `end_date`.
    :param crypto_name
    :type crypto_name: str
    :param end_date
    :type end_date: date
    :return:
    """
    current_date = parse_date('today')
    if end_date >= current_date:
        raise ValueError('end_date must be before today!')

    data_file = os.path.join(MASTER_DATA_DIR, "new_{}.csv".format(CRYPTOCURRENCIES[crypto_name]))
    df = get_historical_df(data_file)
    last_date = df.index[-1].date()
    if last_date >= end_date:
        logger.info("New data is already updated, skipping...")
        return

    logger.info("Updating new {} data for {}...".format(crypto_name, end_date))

    start_date = last_date + timedelta(1)
    new_df = fetch_crypto_data(crypto_name, start_date, end_date)
    updated_df = pd.concat([df, new_df])
    updated_df.to_csv(data_file, index=True, index_label='Date', sep='\t')

    return updated_df
