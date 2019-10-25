"""
Unit tests for the preprocessing.py module.
"""

import os
import pandas as pd
import unittest
from cryptonalysis.config import load_cryptonalysis_config
from cryptonalysis.ml_core.market import market
from cryptonalysis.ml_core.preprocessing import get_historical_df, get_aggregated_dfs, build_transactions_df, \
    get_price_list, normalize_df, normalize_df_with_scaler, load_preprocessed_data, save_preprocessed_data
from cryptonalysis.ml_core.transaction_builders import BiffPredictor, get_transaction_type
from datetime import date
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RES_DIR = 'tests/resources'
CONFIG_TEST = 'config_test.conf'
HISTORICAL_TEST_DATA = 'historical_test_data.csv'
HISTORICAL_TEST_FILE = os.path.join(RES_DIR, HISTORICAL_TEST_DATA)


# Constants for tests
CRYPTO_NAME = 'ETH'
PRICE_COLUMN = 'Close'
PREDICTOR_CLS = BiffPredictor
LOOKAHEAD_DAYS = 1
STARTING_DATE = date(2019, 1, 1)
ENDING_DATE = date(2019, 1, 20)
WINDOW_SIZE = 5
NORMALIZE = True
NORMALIZE_BY_ROW = True
STANDARDIZE = False
PROB_BUY = 1
PROB_SELL = 1


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
            PREDICTOR_CLS(market, price_list=price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME)
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
            PREDICTOR_CLS(market, price_list=price_list, window_size=10, crypto_name=CRYPTO_NAME, lookahead_days=5)
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
            PREDICTOR_CLS(market, price_list=price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME)
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
        normalized_transactions_column, minmax_scaler = normalize_df(transactions, by_row=False, standardize=False)
        self.assertAlmostEqual(normalized_transactions_column.iloc[12, 0], 0, 1)
        self.assertAlmostEqual(normalized_transactions_column.iloc[5, 0], 1, 1)
        self.assertEqual(minmax_scaler.data_min_[0], 116.90)
        self.assertEqual(minmax_scaler.data_max_[0], 157.75)
        standardized_transactions_column, std_scaler = normalize_df(transactions, by_row=False, standardize=True)
        self.assertAlmostEqual(standardized_transactions_column.iloc[0, 0], -0.01774, 5)
        self.assertAlmostEqual(standardized_transactions_column.iloc[14, 0], -1.37288, 5)
        self.assertAlmostEqual(std_scaler.mean_[0], 141.07, 2)
        self.assertAlmostEqual(std_scaler.var_[0], 192.26, 2)

        # Test normalization with scaler
        normalized_transactions = normalize_df_with_scaler(transactions, minmax_scaler)
        self.assertTrue(normalized_transactions.equals(normalized_transactions_column))
        standardized_transactions = normalize_df_with_scaler(transactions, std_scaler)
        self.assertTrue(standardized_transactions.equals(standardized_transactions_column))
        with self.assertRaises(ValueError):
            normalize_df_with_scaler(transactions, MinMaxScaler())
        with self.assertRaises(ValueError):
            normalize_df_with_scaler(transactions, StandardScaler())

    def test_save_and_load_data_file(self):
        # Get transactions DF
        df = get_historical_df(HISTORICAL_TEST_FILE)
        price_list = get_price_list(df, PRICE_COLUMN)

        default_predictor = \
            PREDICTOR_CLS(market, price_list=price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME)
        default_predictor.run_predictor()
        transactions = build_transactions_df(default_predictor.transactions)

        # Test original data
        save_preprocessed_data(transactions, crypto_name=CRYPTO_NAME, predictor_class=PREDICTOR_CLS,
                               lookahead_days=LOOKAHEAD_DAYS, starting_date=STARTING_DATE, ending_date=ENDING_DATE,
                               window_size=WINDOW_SIZE, normalize=False, normalize_by_row=NORMALIZE_BY_ROW,
                               standardize=STANDARDIZE, prob_buy=PROB_BUY, prob_sell=PROB_SELL)
        preprocessed_data = load_preprocessed_data(crypto_name=CRYPTO_NAME, predictor_class=PREDICTOR_CLS,
                                                   lookahead_days=LOOKAHEAD_DAYS, starting_date=STARTING_DATE,
                                                   ending_date=ENDING_DATE, window_size=WINDOW_SIZE,
                                                   normalize=False, normalize_by_row=NORMALIZE_BY_ROW,
                                                   standardize=STANDARDIZE, prob_buy=PROB_BUY, prob_sell=PROB_SELL)
        self.assertTrue(preprocessed_data.equals(transactions))

        # Test normalized data
        normalized_transactions, _ = normalize_df(transactions, NORMALIZE_BY_ROW, STANDARDIZE)
        save_preprocessed_data(normalized_transactions, crypto_name=CRYPTO_NAME, predictor_class=PREDICTOR_CLS,
                               lookahead_days=LOOKAHEAD_DAYS, starting_date=STARTING_DATE, ending_date=ENDING_DATE,
                               window_size=WINDOW_SIZE, normalize=True, normalize_by_row=NORMALIZE_BY_ROW,
                               standardize=STANDARDIZE, prob_buy=PROB_BUY, prob_sell=PROB_SELL)
        normalized_data = load_preprocessed_data(crypto_name=CRYPTO_NAME, predictor_class=PREDICTOR_CLS,
                                                 lookahead_days=LOOKAHEAD_DAYS, starting_date=STARTING_DATE,
                                                 ending_date=ENDING_DATE, window_size=WINDOW_SIZE, normalize=True,
                                                 normalize_by_row=NORMALIZE_BY_ROW, standardize=STANDARDIZE,
                                                 prob_buy=PROB_BUY, prob_sell=PROB_SELL)
        self.assertEqual(normalized_transactions.shape, normalized_data.shape)
        # Data is stored with limited decimals
        for row in range(preprocessed_data.shape[0]):
            for col in range(preprocessed_data.shape[1]):
                self.assertAlmostEqual(normalized_data.iloc[row, col], normalized_transactions.iloc[row, col], 5)
