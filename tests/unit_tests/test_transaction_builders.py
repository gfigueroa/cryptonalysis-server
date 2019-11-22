"""
Unit tests for the transaction_builders.py module.
"""

import pandas as pd
import unittest
from cryptonalysis.ml_core.market import market
from cryptonalysis.ml_core.transaction_builders import BiffPredictor, BiffPredictorSmart, ReverseBiffPredictor
from datetime import date
from pandas import Series


# Constants for tests
CRYPTO_NAME = 'ETH'
DEFAULT_PREDICTOR_CLS = BiffPredictor
WINDOW_SIZE = 5
STARTING_DATE = date(2019, 1, 1)
ENDING_DATE = date(2019, 1, 20)
DATES = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-05', '2019-01-06', '2019-01-07', '2019-01-08',
         '2019-01-09', '2019-01-10', '2019-01-11', '2019-01-12', '2019-01-13', '2019-01-14', '2019-01-15', '2019-01-16',
         '2019-01-17', '2019-01-18', '2019-01-19', '2019-01-20']
PRICES = [10, 20, 30, 20, 20, 10, 20, 30, 40, 50, 40, 50, 60, 70, 80, 70, 60, 60, 50, 60]
PRICE_SERIES = Series(PRICES, DATES)
PRICE_SERIES.index = pd.to_datetime(PRICE_SERIES.index)


class TestCryptoPredictor(unittest.TestCase):
    """
    Test CryptoPredictor superclass methods, independent of the subclass implementations.
    """

    def __init__(self, *args, **kwargs):
        super(TestCryptoPredictor, self).__init__(*args, **kwargs)
        self.default_predictor = \
            DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                                  starting_date=STARTING_DATE, ending_date=ENDING_DATE)

    def test_idempotent_crypto_predictor_methods(self):
        """
        Test idempotent methods in the CryptoPredictor class.
        These tests are done with the initial conditions of the class.
        """
        self.default_predictor.reset_predictor_state()

        # Max crypto transactions
        max_crypto_transaction_buy = self.default_predictor.get_max_crypto_transaction(True, 10)
        # crypto_amount = (self.cash / current_price) - ((self.cash / current_price) * transaction_fee_perc)
        #               = self.cash / (current_price + (current_price * transaction_fee_perc))
        #               = 100.00 / (10.00 + (10.00 * 0.01)) = 9.900990099009901
        self.assertAlmostEqual(max_crypto_transaction_buy, 9.90099, 5)

        # No crypto
        max_crypto_transaction_sell = self.default_predictor.get_max_crypto_transaction(False, 10)
        # crypto_amount = self.owned_crypto
        #               = 0
        self.assertEqual(max_crypto_transaction_sell, 0)
        # Add some crypto
        self.default_predictor.owned_crypto += 10
        max_crypto_transaction_sell = self.default_predictor.get_max_crypto_transaction(False, 10)
        self.assertEqual(max_crypto_transaction_sell, 10)
        self.default_predictor.reset_predictor_state()

        # Buy/Sell all possible
        crypto_amount_buy, fiat_amount_buy = self.default_predictor.buy(10)
        self.assertAlmostEqual(crypto_amount_buy, 9.90099, 5)
        self.assertEqual(fiat_amount_buy, 100)
        # No crypto
        crypto_amount_sell, fiat_amount_sell = self.default_predictor.sell(10)
        self.assertEqual(crypto_amount_sell, 0)
        self.assertEqual(fiat_amount_sell, 0)
        # Add some crypto
        self.default_predictor.owned_crypto += 10
        crypto_amount_sell, fiat_amount_sell = self.default_predictor.sell(10)
        self.assertEqual(crypto_amount_sell, 10)
        self.assertEqual(fiat_amount_sell, 99)
        self.default_predictor.reset_predictor_state()

        # Transaction fiat amounts
        transaction_fiat_amount_buy = self.default_predictor.get_transaction_fiat_amount(True, 1, 10)
        # amount_buy = (crypto_amount * current_price) + (crypto_amount * current_price * transaction_fee_perc)
        #            = (1 * 10.00) + (1 * 10.00 * 0.01) = 10.10
        self.assertEqual(transaction_fiat_amount_buy, 10.1)
        transaction_fiat_amount_buy = self.default_predictor.get_transaction_fiat_amount(True, 10, 10)
        # amount_buy = (crypto_amount * current_price) + (crypto_amount * current_price * transaction_fee_perc)
        #            = (10 * 10.00) + (10 * 10.00 * 0.01) = 101
        self.assertEqual(transaction_fiat_amount_buy, 101)

        transaction_fiat_amount_sell = self.default_predictor.get_transaction_fiat_amount(False, 1, 10)
        # amount_sell = (crypto_amount * current_price) - (crypto_amount * current_price * transaction_fee_perc)
        #             = (1 * 10.00) - (1 * 10.00 * 0.01) = 9.9
        self.assertEqual(transaction_fiat_amount_sell, 9.9)
        transaction_fiat_amount_sell = self.default_predictor.get_transaction_fiat_amount(False, 10, 10)
        # amount_sell = (crypto_amount * current_price) - (crypto_amount * current_price * transaction_fee_perc)
        #             = (10 * 10.00) - (10 * 10.00 * 0.01) = 99
        self.assertEqual(transaction_fiat_amount_sell, 99)

    def test_default_crypto_predictor_methods(self):
        """
        Test non-idempotent (with effects on state variables) methods in the CryptoPredictor class with the default
        parameters.
        These tests are done with the initial conditions of the class.
        """
        self.default_predictor.reset_predictor_state()

        # Run Predictor
        self.default_predictor.run_predictor()
        self.assertEqual(self.default_predictor.owned_crypto, 0)  # All crypto sold at the end
        self.assertGreater(self.default_predictor.cash, 0)
        self.assertEqual(self.default_predictor.total_investment, 175)
        self.assertEqual(len(self.default_predictor.transactions), 15)

        # Transaction Simulation (same for all implementations unless overridden)
        transactions = [1, 0]
        self.default_predictor.run_transaction_simulation(transactions)
        self.assertEqual(self.default_predictor.owned_crypto, 0)
        self.assertAlmostEqual(self.default_predictor.cash, 59.00990, 5)
        transactions = [0, 1]
        self.default_predictor.run_transaction_simulation(transactions)
        self.assertEqual(self.default_predictor.owned_crypto, 0)
        self.assertAlmostEqual(self.default_predictor.cash, 105.97981, 5)

        # Verify that run_predictor() and run_transaction_simulation() have same results
        actual_run = DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE,
                                           crypto_name=CRYPTO_NAME, starting_date=STARTING_DATE,
                                           ending_date=ENDING_DATE)
        actual_run.run_predictor()
        actual_transactions = [t['transaction'] for t in actual_run.transactions]
        simulation_run = DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE,
                                               crypto_name=CRYPTO_NAME, starting_date=STARTING_DATE,
                                               ending_date=ENDING_DATE)
        simulation_run.run_transaction_simulation(actual_transactions)
        self.assertEqual(actual_run.owned_crypto, simulation_run.owned_crypto)
        self.assertEqual(actual_run.cash, simulation_run.cash)
        self.assertEqual(actual_run.total_investment, simulation_run.total_investment)

    def test_modified_crypto_predictor_methods(self):
        """
        Test non-idempotent (with effects on state variables) methods in the CryptoPredictor class with different
        parameters.
        """

        # Most transactions possible
        predictor = DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=1, crypto_name=CRYPTO_NAME,
                                          starting_date=STARTING_DATE, ending_date=ENDING_DATE, lookahead_days=1)
        predictor.run_predictor()
        self.assertEqual(predictor.total_investment, 195)
        self.assertEqual(len(predictor.transactions), 19)

        # Least transactions possible
        predictor = DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=1, crypto_name=CRYPTO_NAME,
                                          starting_date=STARTING_DATE, ending_date=ENDING_DATE, lookahead_days=19)
        predictor.run_predictor()
        self.assertEqual(predictor.total_investment, 105)
        self.assertEqual(len(predictor.transactions), 1)

        # Wrong params
        with self.assertRaises(ValueError):
            DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=0, crypto_name=CRYPTO_NAME,
                                  starting_date=STARTING_DATE, ending_date=ENDING_DATE)
        with self.assertRaises(ValueError):
            DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=20, crypto_name=CRYPTO_NAME,
                                  starting_date=STARTING_DATE, ending_date=ENDING_DATE)
        with self.assertRaises(ValueError):
            DEFAULT_PREDICTOR_CLS(market, price_list=PRICE_SERIES, window_size=1, crypto_name=CRYPTO_NAME,
                                  starting_date=STARTING_DATE, ending_date=ENDING_DATE, lookahead_days=20)


class TestCryptoPredictorImplementations(unittest.TestCase):
    """
    Test CryptoPredictor implementations and their functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestCryptoPredictorImplementations, self).__init__(*args, **kwargs)
        self.biff_predictor = \
            BiffPredictor(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                          starting_date=STARTING_DATE, ending_date=ENDING_DATE)
        self.biff_predictor_smart = \
            BiffPredictorSmart(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                               starting_date=STARTING_DATE, ending_date=ENDING_DATE)
        self.reverse_biff_predictor = \
            ReverseBiffPredictor(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE, crypto_name=CRYPTO_NAME,
                                 starting_date=STARTING_DATE, ending_date=ENDING_DATE)

    def test_biff_predictor(self):
        """
        Test non-idempotent methods in the CryptoPredictor class (with effects on state variables).
        These tests are done with the initial conditions of the class.
        """
        self.biff_predictor.reset_predictor_state()

        # Run Predictor with default parameters
        self.biff_predictor.run_predictor()
        self.assertEqual(self.biff_predictor.owned_crypto, 0)  # All crypto sold at the end
        self.assertAlmostEqual(self.biff_predictor.cash, 1089.51665, 5)
        self.assertEqual(self.biff_predictor.total_investment, 175)
        self.assertEqual(len(self.biff_predictor.transactions), 15)
        actual_transactions = [t['transaction'] for t in self.biff_predictor.transactions]
        expected_transactions = ['SELL', 'BUY', 'BUY', 'BUY', 'BUY', 'SELL', 'BUY', 'BUY', 'BUY', 'BUY', 'SELL', 'SELL',
                                 'SELL', 'SELL', 'BUY']
        self.assertListEqual(actual_transactions, expected_transactions)

        # Run Predictor with lookahead days=2
        modified_biff_predictor = BiffPredictor(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE,
                                                crypto_name=CRYPTO_NAME, starting_date=STARTING_DATE,
                                                ending_date=ENDING_DATE, lookahead_days=2)
        modified_biff_predictor.run_predictor()
        self.assertEqual(modified_biff_predictor.owned_crypto, 0)  # All crypto sold at the end
        self.assertAlmostEqual(modified_biff_predictor.cash, 789.19383, 5)
        self.assertEqual(modified_biff_predictor.total_investment, 170)
        self.assertEqual(len(modified_biff_predictor.transactions), 14)
        actual_transactions = [t['transaction'] for t in modified_biff_predictor.transactions]
        expected_transactions = ['SELL', 'BUY', 'BUY', 'BUY', 'SELL', 'SELL', 'BUY', 'BUY', 'BUY', 'SELL', 'SELL',
                                 'SELL', 'SELL', 'SELL']
        self.assertListEqual(actual_transactions, expected_transactions)

    def test_biff_predictor_smart(self):
        """
        Test non-idempotent methods in the CryptoPredictor class (with effects on state variables).
        These tests are done with the initial conditions of the class.
        """
        self.biff_predictor_smart.reset_predictor_state()

        # Run Predictor with default parameters
        self.biff_predictor_smart.run_predictor()
        self.assertEqual(self.biff_predictor_smart.owned_crypto, 0)  # All crypto sold at the end
        self.assertAlmostEqual(self.biff_predictor_smart.cash, 1089.51665, 5)
        self.assertEqual(self.biff_predictor_smart.total_investment, 175)
        self.assertEqual(len(self.biff_predictor_smart.transactions), 15)
        actual_transactions = [t['transaction'] for t in self.biff_predictor_smart.transactions]
        expected_transactions = ['SELL', 'BUY', 'BUY', 'BUY', 'BUY', 'SELL', 'BUY', 'BUY', 'BUY', 'BUY', 'SELL', 'SELL',
                                 'SELL', 'SELL', 'BUY']
        self.assertListEqual(actual_transactions, expected_transactions)

        # Run Predictor with lookahead days=2
        modified_biff_predictor_smart = BiffPredictorSmart(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE,
                                                           crypto_name=CRYPTO_NAME, starting_date=STARTING_DATE,
                                                           ending_date=ENDING_DATE, lookahead_days=2)
        modified_biff_predictor_smart.run_predictor()
        self.assertEqual(modified_biff_predictor_smart.owned_crypto, 0)  # All crypto sold at the end
        self.assertAlmostEqual(modified_biff_predictor_smart.cash, 1106.42608, 5)
        self.assertEqual(modified_biff_predictor_smart.total_investment, 170)
        self.assertEqual(len(modified_biff_predictor_smart.transactions), 14)
        actual_transactions = [t['transaction'] for t in modified_biff_predictor_smart.transactions]
        expected_transactions = ['SELL', 'BUY', 'BUY', 'BUY', 'BUY', 'SELL', 'BUY', 'BUY', 'BUY', 'BUY', 'SELL', 'SELL',
                                 'SELL', 'SELL']
        self.assertListEqual(actual_transactions, expected_transactions)

    def test_reverse_biff_predictor(self):
        """
        Test non-idempotent methods in the CryptoPredictor class (with effects on state variables).
        These tests are done with the initial conditions of the class.
        """
        self.reverse_biff_predictor.reset_predictor_state()

        # Run Predictor with default parameters
        self.reverse_biff_predictor.run_predictor()
        self.assertEqual(self.reverse_biff_predictor.owned_crypto, 0)  # All crypto sold at the end
        self.assertAlmostEqual(self.reverse_biff_predictor.cash, 72.53854, 5)
        self.assertEqual(self.reverse_biff_predictor.total_investment, 175)
        self.assertEqual(len(self.reverse_biff_predictor.transactions), 15)
        actual_transactions = [t['transaction'] for t in self.reverse_biff_predictor.transactions]
        expected_transactions = ['BUY', 'SELL', 'SELL', 'SELL', 'SELL', 'BUY', 'SELL', 'SELL', 'SELL', 'SELL', 'BUY',
                                 'BUY', 'BUY', 'BUY', 'SELL']
        self.assertListEqual(actual_transactions, expected_transactions)

        # Run Predictor with lookahead days=2
        modified_reverse_biff_predictor = ReverseBiffPredictor(market, price_list=PRICE_SERIES, window_size=WINDOW_SIZE,
                                                               crypto_name=CRYPTO_NAME, starting_date=STARTING_DATE,
                                                               ending_date=ENDING_DATE, lookahead_days=2)
        modified_reverse_biff_predictor.run_predictor()
        self.assertEqual(modified_reverse_biff_predictor.owned_crypto, 0)  # All crypto sold at the end
        self.assertAlmostEqual(modified_reverse_biff_predictor.cash, 99.60801, 5)
        self.assertEqual(modified_reverse_biff_predictor.total_investment, 170)
        self.assertEqual(len(modified_reverse_biff_predictor.transactions), 14)
        actual_transactions = [t['transaction'] for t in modified_reverse_biff_predictor.transactions]
        expected_transactions = ['BUY', 'SELL', 'SELL', 'SELL', 'BUY', 'BUY', 'SELL', 'SELL', 'SELL', 'BUY', 'BUY',
                                 'BUY', 'BUY', 'BUY']
        self.assertListEqual(actual_transactions, expected_transactions)
