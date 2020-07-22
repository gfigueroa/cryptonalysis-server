from cryptonalysis.ml_core.preprocessing import CRYPTOCURRENCIES
from cryptonalysis.utils.misc_utils import parse_date
from cryptonalysis.utils.data_link import update_new_data


if __name__ == '__main__':
    # Update new data
    for crypto in CRYPTOCURRENCIES.keys():
        yesterday = parse_date('yesterday')
        update_new_data(crypto, yesterday)
