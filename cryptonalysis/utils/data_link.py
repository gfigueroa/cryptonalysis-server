import pandas as pd
from datetime import date
from cryptonalysis.ml_core.preprocessing import CRYPTOCURRENCIES
from misc_utils import parse_date


API_URLS = {
    crypto_short: "https://coinmarketcap.com/currencies/{}/historical-data/".format(crypto_long)
    for crypto_short, crypto_long in CRYPTOCURRENCIES.items()
}


def fetch_crypto_data(crypto, start_date, end_date):
    """

    :param crypto
    :type crypto: str
    :param start_date
    :type start_date: date
    :param end_date
    :type end_date: date
    :return:
    """

    # get market info for bitcoin from the start of 2016 to the current day
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    crypto_endpoint = "{}?start={}&end={}".format(API_URLS[crypto], start_date_str, end_date_str)
    bitcoin_market_info = pd.read_html(crypto_endpoint)[0]
    print bitcoin_market_info


if __name__ == '__main__':
    crypto = 'ETH'
    start = parse_date('yesterday')
    end = parse_date('yesterday')
    fetch_crypto_data(crypto, start, end)
