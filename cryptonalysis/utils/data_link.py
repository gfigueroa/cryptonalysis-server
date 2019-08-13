import pandas as pd
from datetime import date
from cryptonalysis.ml_core.preprocessing import CRYPTOCURRENCIES
from misc_utils import parse_date


API_URLS = {
    crypto_short: "https://coinmarketcap.com/currencies/{}/historical-data/".format(crypto_long)
    for crypto_short, crypto_long in CRYPTOCURRENCIES.items()
}


def fetch_crypto_data(crypto_name, start_date, end_date):
    """
    Fetch cryptocurrency data from a remote API between a given start date and a given end date.
    :param crypto_name
    :type crypto_name: str
    :param start_date
    :type start_date: date
    :param end_date
    :type end_date: date
    :return: A cleansed DataFrame with cryptocurrency data.
    :rtype: pd.DataFrame
    """

    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    crypto_endpoint = "{}?start={}&end={}".format(API_URLS[crypto_name], start_date_str, end_date_str)

    df = pd.read_html(crypto_endpoint)[0]

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

    return df


if __name__ == '__main__':
    crypto = 'ETH'
    start = parse_date('2019-08-01')
    end = parse_date('yesterday')
    crypto_data = fetch_crypto_data(crypto, start, end)
    print crypto_data
