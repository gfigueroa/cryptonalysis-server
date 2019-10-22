"""
Unit tests for the preprocessing.py module.
"""

import os
import pandas as pd
import unittest
from cryptonalysis.config import load_cryptonalysis_config
from cryptonalysis.ml_core.market import market
from cryptonalysis.ml_core.preprocessing import get_historical_df, get_aggregated_dfs, build_transactions_df, \
    get_price_list, normalize_df
from cryptonalysis.ml_core.transaction_builders import BiffPredictor, get_transaction_type

RES_DIR = 'tests/resources'
CONFIG_TEST = 'config_test.conf'
HISTORICAL_TEST_DATA = 'historical_test_data.csv'
HISTORICAL_TEST_FILE = os.path.join(RES_DIR, HISTORICAL_TEST_DATA)


# Constants for tests
CRYPTO_NAME = 'ETH'
WINDOW_SIZE = 5
PRICE_COLUMN = 'Close'


class TestPreprocessingFunctions(unittest.TestCase):
    """
    Test CryptoPredictor superclass methods, independent of the subclass implementations.
    """

    def __init__(self, *args, **kwargs):
        super(TestPreprocessingFunctions, self).__init__(*args, **kwargs)
        self.config = load_cryptonalysis_config(RES_DIR, CONFIG_TEST)

    def test_get_historical_df(self):
        df = get_historical_df(HISTORICAL_TEST_FILE)
        real_columns = list(df.columns)
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
        self.assertListEqual(real_columns, expected_columns)
        self.assertEqual(df.shape, (20, 7))
        self.assertEqual(df['Date'].iloc[0], pd.Timestamp('2019-01-01 00:00:00'))
        self.assertEqual(df['Date'].iloc[-1], pd.Timestamp('2019-01-20 00:00:00'))

    def test_get_aggregated_dfs(self):
        df = get_historical_df(HISTORICAL_TEST_FILE)
        daily_df, weekly_df, monthly_df = get_aggregated_dfs(df)
        self.assertEqual(daily_df.shape, (20, 6))
        self.assertEqual(weekly_df.shape, (3, 6))
        self.assertEqual(monthly_df.shape, (1, 6))

    def test_build_transactions_df(self):
        df = get_historical_df(HISTORICAL_TEST_FILE)
        price_list = get_price_list(df, PRICE_COLUMN)

        # Test default predictor
        default_predictor = \
            BiffPredictor(market, price_list=price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME)
        default_predictor.run_predictor()
        transactions = build_transactions_df(default_predictor.transactions)
        self.assertEqual(transactions.shape, (15, 6))
        self.assertEqual(transactions.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(transactions.index[-1], pd.Timestamp('2019-01-19 00:00:00'))
        expected_transactions = ['BUY', 'SELL', 'SELL', 'BUY', 'SELL', 'SELL', 'SELL', 'SELL', 'BUY', 'SELL', 'BUY',
                                 'BUY', 'SELL', 'BUY', 'SELL']
        actual_transactions = [get_transaction_type(t) for t in list(transactions['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)

        # Test predictor with different params
        modified_predictor = \
            BiffPredictor(market, price_list=price_list, window_size=10, crypto_name=CRYPTO_NAME, lookahead_days=5)
        modified_predictor.run_predictor()
        transactions = build_transactions_df(modified_predictor.transactions)
        self.assertEqual(transactions.shape, (6, 11))
        self.assertEqual(transactions.index[0], pd.Timestamp('2019-01-10 00:00:00'))
        self.assertEqual(transactions.index[-1], pd.Timestamp('2019-01-15 00:00:00'))
        expected_transactions = ['SELL', 'SELL', 'SELL', 'BUY', 'SELL', 'SELL']
        actual_transactions = [get_transaction_type(t) for t in list(transactions['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)

    def test_normalize_df(self):
        # Get transactions DF
        df = get_historical_df(HISTORICAL_TEST_FILE)
        price_list = get_price_list(df, PRICE_COLUMN)

        default_predictor = \
            BiffPredictor(market, price_list=price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME)
        default_predictor.run_predictor()
        transactions = build_transactions_df(default_predictor.transactions)

        # Test normalization by row
        normalized_transactions_row, _ = normalize_df(transactions, by_row=True, standardize=False)
        self.assertEqual(normalized_transactions_row.iloc[0, 0], 0)
        self.assertEqual(normalized_transactions_row.iloc[0, 4], 1)
        standardized_transactions_row, _ = normalize_df(transactions, by_row=True, standardize=True)
        self.assertAlmostEqual(standardized_transactions_row.iloc[0, 0], -1.82010, 5)
        self.assertAlmostEqual(standardized_transactions_row.iloc[0, 4], 0.81768, 5)

        # Test normalization by column
        normalized_transactions_column, scaler = normalize_df(transactions, by_row=False, standardize=False)
        self.assertAlmostEqual(normalized_transactions_column.iloc[12, 0], 0, 1)
        self.assertAlmostEqual(normalized_transactions_column.iloc[5, 0], 1, 1)
        standardized_transactions_column, scaler = normalize_df(transactions, by_row=False, standardize=True)
        self.assertAlmostEqual(standardized_transactions_column.iloc[0, 0], -0.01774, 5)
        self.assertAlmostEqual(standardized_transactions_column.iloc[14, 0], -1.37288, 5)
