"""
Unit tests for the preprocessing.py module.
"""

import copy
import os
import pandas as pd
import unittest
from cryptonalysis.config import PreprocessingConfig
from cryptonalysis.ml_core.market import market
from cryptonalysis.ml_core.preprocessing import get_historical_df, build_transactions_df, get_price_list, \
    normalize_df, normalize_df_with_scaler, load_preprocessed_data, save_preprocessed_data, preprocess_dataframe, \
    preprocess_data_point, save_scaler, load_scaler
from cryptonalysis.ml_core.transaction_builders import get_transaction_type,\
    get_predictor_class_from_name
from datetime import date
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RES_DIR = os.path.join('tests', 'resources')
HISTORICAL_TEST_DATA = 'historical_test_data.csv'
HISTORICAL_TEST_FILE = os.path.join(RES_DIR, HISTORICAL_TEST_DATA)


# Constants for tests
CRYPTO_NAME = 'ETH'
STARTING_DATE = date(2019, 1, 1)
ENDING_DATE = date(2019, 1, 20)
WINDOW_SIZE = 5
NORMALIZE = True
NORMALIZE_BY_ROW = True
STANDARDIZE = False
PRICE_COLUMN = 'Close'
PREDICTOR_CLS_NAME = 'BiffPredictor'
PREDICTOR_CLS = get_predictor_class_from_name(PREDICTOR_CLS_NAME)
PROB_BUY = 1
PROB_SELL = 1
STARTING_INVESTMENT = 100
DAILY_ALLOWANCE = 5
LOOKAHEAD_DAYS = 1
PREPROCESSING_CONFIG = PreprocessingConfig({
    'start_date': STARTING_DATE,
    'end_date': ENDING_DATE,
    'window_size': WINDOW_SIZE,
    'normalize': NORMALIZE,
    'normalize_by_row': NORMALIZE_BY_ROW,
    'standardize': STANDARDIZE,
    'price_column': PRICE_COLUMN,
    'predictor_cls': PREDICTOR_CLS_NAME,
    'predictor_params': {
        'prob_buy': PROB_BUY,
        'prob_sell': PROB_SELL,
        'starting_investment': STARTING_INVESTMENT,
        'daily_allowance': DAILY_ALLOWANCE,
        'lookahead_days': LOOKAHEAD_DAYS
    }
})


class TestPreprocessingFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPreprocessingFunctions, self).__init__(*args, **kwargs)
        self.df = get_historical_df(HISTORICAL_TEST_FILE)
        self.price_list = get_price_list(self.df, PRICE_COLUMN)

        # Row-based standardization config
        row_based_standardization_config_dict = copy.deepcopy(PREPROCESSING_CONFIG.config_dict)
        row_based_standardization_config_dict['standardize'] = True
        self.row_based_standardization_config = PreprocessingConfig(row_based_standardization_config_dict)

        # Column-based normalization config
        column_based_normalization_config_dict = copy.deepcopy(PREPROCESSING_CONFIG.config_dict)
        column_based_normalization_config_dict['normalize_by_row'] = False
        self.column_based_normalization_config = PreprocessingConfig(column_based_normalization_config_dict)

        # Column-based standardization config
        column_based_standardization_config_dict = copy.deepcopy(PREPROCESSING_CONFIG.config_dict)
        column_based_standardization_config_dict['normalize_by_row'] = False
        column_based_standardization_config_dict['standardize'] = True
        self.column_based_standardization_config = PreprocessingConfig(column_based_standardization_config_dict)

    def test_get_historical_df(self):
        real_columns = list(self.df.columns)
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
        self.assertListEqual(real_columns, expected_columns)
        self.assertEqual(self.df.shape, (20, 6))
        self.assertEqual(self.df.index[0], pd.Timestamp('2019-01-01 00:00:00'))
        self.assertEqual(self.df.index[-1], pd.Timestamp('2019-01-20 00:00:00'))

    def test_build_transactions_df(self):
        # Test default predictor
        default_predictor = \
            PREDICTOR_CLS(market, price_list=self.price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                          starting_date=STARTING_DATE, ending_date=ENDING_DATE)
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
            PREDICTOR_CLS(market, price_list=self.price_list, window_size=10, crypto_name=CRYPTO_NAME,
                          starting_date=STARTING_DATE, ending_date=ENDING_DATE, lookahead_days=5)
        modified_predictor.run_predictor()
        transactions = build_transactions_df(modified_predictor.transactions)
        self.assertEqual(transactions.shape, (6, 11))
        self.assertEqual(transactions.index[0], pd.Timestamp('2019-01-10 00:00:00'))
        self.assertEqual(transactions.index[-1], pd.Timestamp('2019-01-15 00:00:00'))
        expected_transactions = ['SELL', 'SELL', 'SELL', 'BUY', 'SELL', 'SELL']
        actual_transactions = [get_transaction_type(t) for t in list(transactions['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)

    def test_normalize_df(self):
        # Run predictor
        default_predictor = \
            PREDICTOR_CLS(market, price_list=self.price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                          starting_date=STARTING_DATE, ending_date=ENDING_DATE)
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
        # Run predictor
        default_predictor = \
            PREDICTOR_CLS(market, price_list=self.price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                          starting_date=STARTING_DATE, ending_date=ENDING_DATE)
        default_predictor.run_predictor()
        transactions = build_transactions_df(default_predictor.transactions)

        # Test original data
        save_preprocessed_data(transactions, crypto_name=CRYPTO_NAME, preprocessing_config=PREPROCESSING_CONFIG,
                               preprocessed_data_dir=RES_DIR)
        preprocessed_data = load_preprocessed_data(crypto_name=CRYPTO_NAME, preprocessing_config=PREPROCESSING_CONFIG,
                                                   preprocessed_data_dir=RES_DIR)
        self.assertEqual(transactions.shape, preprocessed_data.shape)
        # Data is stored with limited decimals
        for row in range(transactions.shape[0]):
            for col in range(transactions.shape[1]):
                self.assertAlmostEqual(preprocessed_data.iloc[row, col], transactions.iloc[row, col], 5)

        # Test normalized data
        normalized_transactions, _ = normalize_df(transactions, NORMALIZE_BY_ROW, STANDARDIZE)
        save_preprocessed_data(normalized_transactions, crypto_name=CRYPTO_NAME,
                               preprocessing_config=PREPROCESSING_CONFIG, preprocessed_data_dir=RES_DIR)
        normalized_data = load_preprocessed_data(crypto_name=CRYPTO_NAME, preprocessing_config=PREPROCESSING_CONFIG,
                                                 preprocessed_data_dir=RES_DIR)
        self.assertEqual(normalized_transactions.shape, normalized_data.shape)
        # Data is stored with limited decimals
        for row in range(normalized_transactions.shape[0]):
            for col in range(normalized_transactions.shape[1]):
                self.assertAlmostEqual(normalized_data.iloc[row, col], normalized_transactions.iloc[row, col], 5)

    def test_save_and_load_scaler(self):
        # Run predictor
        default_predictor = \
            PREDICTOR_CLS(market, price_list=self.price_list, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                          starting_date=STARTING_DATE, ending_date=ENDING_DATE)
        default_predictor.run_predictor()
        transactions = build_transactions_df(default_predictor.transactions)

        # Normalize by column
        normalized_transactions_column, minmax_scaler = normalize_df(transactions, by_row=False, standardize=False)
        save_scaler(minmax_scaler, CRYPTO_NAME, self.column_based_normalization_config, scaler_dir=RES_DIR)
        saved_scaler = load_scaler(CRYPTO_NAME, self.column_based_normalization_config, scaler_dir=RES_DIR)
        self.assertEqual(minmax_scaler.data_min_[0], saved_scaler.data_min_[0])
        self.assertEqual(minmax_scaler.data_max_[0], saved_scaler.data_max_[0])

        # Standardize by column
        standardized_transactions_column, std_scaler = normalize_df(transactions, by_row=False, standardize=True)
        save_scaler(std_scaler, CRYPTO_NAME, self.column_based_standardization_config, scaler_dir=RES_DIR)
        saved_scaler = load_scaler(CRYPTO_NAME, self.column_based_standardization_config, scaler_dir=RES_DIR)
        self.assertEqual(std_scaler.mean_[0], saved_scaler.mean_[0])
        self.assertEqual(std_scaler.var_[0], saved_scaler.var_[0])

    def test_preprocess_dataframe(self):
        expected_transactions = ['BUY', 'SELL', 'SELL', 'BUY', 'SELL', 'SELL', 'SELL', 'SELL', 'BUY', 'SELL', 'BUY',
                                 'BUY', 'SELL', 'BUY', 'SELL']

        # Run default preprocessing (row-based normalization)
        transactions_df = \
            preprocess_dataframe(self.df, crypto_name=CRYPTO_NAME, preprocessing_config=PREPROCESSING_CONFIG,
                                 scaler_dir=RES_DIR)
        self.assertEquals(transactions_df.shape, (15, 6))
        self.assertEqual(transactions_df.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(transactions_df.index[-1], pd.Timestamp('2019-01-19 00:00:00'))
        actual_transactions = [get_transaction_type(t) for t in list(transactions_df['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)
        self.assertEqual(transactions_df.iloc[0, 0], 0)
        self.assertEqual(transactions_df.iloc[0, 4], 1)
        self.assertEqual(transactions_df.iloc[14, 3], 0)
        self.assertEqual(transactions_df.iloc[14, 4], 1)

        # Run row-based standardization
        transactions_df = \
            preprocess_dataframe(self.df, crypto_name=CRYPTO_NAME,
                                 preprocessing_config=self.row_based_standardization_config, scaler_dir=RES_DIR)
        self.assertEquals(transactions_df.shape, (15, 6))
        self.assertEqual(transactions_df.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(transactions_df.index[-1], pd.Timestamp('2019-01-19 00:00:00'))
        actual_transactions = [get_transaction_type(t) for t in list(transactions_df['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)
        self.assertAlmostEquals(transactions_df.iloc[0, 0], -1.82010, 5)
        self.assertAlmostEquals(transactions_df.iloc[0, 4], 0.81768, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 0], -0.74050, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 4], 1.22104, 5)

        # Run preprocessing with column-based normalization
        transactions_df = \
            preprocess_dataframe(self.df, crypto_name=CRYPTO_NAME,
                                 preprocessing_config=self.column_based_normalization_config, scaler_dir=RES_DIR)
        self.assertTrue(os.path.exists(os.path.join(  # Make sure scaler was created
            RES_DIR, 'scaler_ETH_2019-01-20TrueFalseBiffPredictor5111100CloseFalse2019-01-015.joblib')))
        self.assertEquals(transactions_df.shape, (15, 6))
        self.assertEqual(transactions_df.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(transactions_df.index[-1], pd.Timestamp('2019-01-19 00:00:00'))
        actual_transactions = [get_transaction_type(t) for t in list(transactions_df['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)
        self.assertAlmostEquals(transactions_df.iloc[0, 0], 0.58556, 5)
        self.assertAlmostEquals(transactions_df.iloc[0, 4], 0.94835, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 0], 0.12558, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 4], 0.18654, 5)

        # Run preprocessing with column-based standardization
        transactions_df = \
            preprocess_dataframe(self.df, crypto_name=CRYPTO_NAME,
                                 preprocessing_config=self.column_based_standardization_config, scaler_dir=RES_DIR)
        self.assertTrue(os.path.exists(os.path.join(  # Make sure scaler was created
            RES_DIR, 'scaler_ETH_2019-01-20TrueFalseBiffPredictor5111100CloseTrue2019-01-015.joblib')))
        self.assertEquals(transactions_df.shape, (15, 6))
        self.assertEqual(transactions_df.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(transactions_df.index[-1], pd.Timestamp('2019-01-19 00:00:00'))
        actual_transactions = [get_transaction_type(t) for t in list(transactions_df['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)
        self.assertAlmostEquals(transactions_df.iloc[0, 0], -0.01774, 5)
        self.assertAlmostEquals(transactions_df.iloc[0, 4], 1.54328, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 0], -1.37288, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 4], -0.67075, 5)

        # Run preprocessing for predicting with column-based normalization
        transactions_df = \
            preprocess_dataframe(self.df, crypto_name=CRYPTO_NAME,
                                 preprocessing_config=self.column_based_normalization_config, predicting=True,
                                 scaler_dir=RES_DIR)
        self.assertEquals(transactions_df.shape, (15, 6))
        self.assertEqual(transactions_df.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(transactions_df.index[-1], pd.Timestamp('2019-01-19 00:00:00'))
        actual_transactions = [get_transaction_type(t) for t in list(transactions_df['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)
        self.assertAlmostEquals(transactions_df.iloc[0, 0], 0.58556, 5)
        self.assertAlmostEquals(transactions_df.iloc[0, 4], 0.94835, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 0], 0.12558, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 4], 0.18654, 5)

        # Run preprocessing for predicting with column-based standardization
        transactions_df = \
            preprocess_dataframe(self.df, crypto_name=CRYPTO_NAME,
                                 preprocessing_config=self.column_based_standardization_config, predicting=True,
                                 scaler_dir=RES_DIR)
        self.assertEquals(transactions_df.shape, (15, 6))
        self.assertEqual(transactions_df.index[0], pd.Timestamp('2019-01-05 00:00:00'))
        self.assertEqual(transactions_df.index[-1], pd.Timestamp('2019-01-19 00:00:00'))
        actual_transactions = [get_transaction_type(t) for t in list(transactions_df['transaction'])]
        self.assertListEqual(expected_transactions, actual_transactions)
        self.assertAlmostEquals(transactions_df.iloc[0, 0], -0.01774, 5)
        self.assertAlmostEquals(transactions_df.iloc[0, 4], 1.54328, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 0], -1.37288, 5)
        self.assertAlmostEquals(transactions_df.iloc[14, 4], -0.67075, 5)

    def test_preprocess_data_point(self):
        # Run default preprocessing
        transaction_df = \
            preprocess_data_point(self.df, crypto_name=CRYPTO_NAME, preprocessing_config=PREPROCESSING_CONFIG,
                                  scaler_dir=RES_DIR)
        self.assertEquals(transaction_df.shape, (1, 6))
        self.assertEqual(transaction_df.index[0], pd.Timestamp('2019-01-20 00:00:00'))
        self.assertEqual(transaction_df.iloc[0, 4], 0)
        self.assertEqual(transaction_df.iloc[0, 3], 1)
        expected_transaction = ['UNKNOWN']
        actual_transaction = [get_transaction_type(t) for t in list(transaction_df['transaction'])]
        self.assertListEqual(expected_transaction, actual_transaction)

        # Run default preprocessing with column-based normalization
        transaction_df = \
            preprocess_data_point(self.df, crypto_name=CRYPTO_NAME,
                                  preprocessing_config=self.column_based_normalization_config, scaler_dir=RES_DIR)
        self.assertEquals(transaction_df.shape, (1, 6))
        self.assertEqual(transaction_df.index[0], pd.Timestamp('2019-01-20 00:00:00'))
        self.assertAlmostEquals(transaction_df.iloc[0, 0], 0.16279, 5)
        self.assertAlmostEquals(transaction_df.iloc[0, 4], 0.06291, 5)
        expected_transaction = ['UNKNOWN']
        actual_transaction = [get_transaction_type(t) for t in list(transaction_df['transaction'])]
        self.assertListEqual(expected_transaction, actual_transaction)

        # Run default preprocessing with column-based standardization
        transaction_df = \
            preprocess_data_point(self.df, crypto_name=CRYPTO_NAME,
                                  preprocessing_config=self.column_based_standardization_config, scaler_dir=RES_DIR)
        self.assertEquals(transaction_df.shape, (1, 6))
        self.assertEqual(transaction_df.index[0], pd.Timestamp('2019-01-20 00:00:00'))
        self.assertAlmostEquals(transaction_df.iloc[0, 0], -1.26325, 5)
        self.assertAlmostEquals(transaction_df.iloc[0, 4], -1.03004, 5)
        expected_transaction = ['UNKNOWN']
        actual_transaction = [get_transaction_type(t) for t in list(transaction_df['transaction'])]
        self.assertListEqual(expected_transaction, actual_transaction)
