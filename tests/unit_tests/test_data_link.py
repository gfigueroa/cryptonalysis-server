"""
Unit tests for the data_link.py module.
"""

import pandas as pd
import unittest
from cryptonalysis.utils.data_link import fetch_crypto_data, get_crypto_data_for_date
from cryptonalysis.utils.misc_utils import parse_date
from datetime import date


# Constants for tests
CRYPTO_NAME = 'ETH'
STARTING_DATE = date(2019, 1, 1)
ENDING_DATE = date(2019, 1, 20)
FOR_DATE = parse_date('2019-02-01')
WINDOW_SIZE = 5


class TestPredictionFunctions(unittest.TestCase):
    def test_fetch_crypto_data(self):
        # Test good case
        data = fetch_crypto_data(CRYPTO_NAME, STARTING_DATE, ENDING_DATE)
        self.assertEqual(data.shape, (20, 6))
        self.assertEqual(data.index[0], pd.Timestamp(STARTING_DATE))
        self.assertEqual(data.index[-1], pd.Timestamp(ENDING_DATE))
        self.assertEquals(data['Close'].iloc[0], 140.82)
        self.assertEquals(data['Close'].iloc[-1], 119.47)

        # Test bad case
        with self.assertRaises(ValueError):
            fetch_crypto_data(CRYPTO_NAME, ENDING_DATE, STARTING_DATE)

    def test_get_crypto_data_for_date(self):
        # Test good case
        data = get_crypto_data_for_date(CRYPTO_NAME, FOR_DATE, WINDOW_SIZE)
        self.assertEqual(data.shape, (5, 6))
        self.assertEqual(data.index[0], pd.Timestamp('2019-01-27 00:00:00'))
        self.assertEqual(data.index[-1], pd.Timestamp('2019-01-31 00:00:00'))
        self.assertEquals(data['Close'].iloc[0], 113.41)
        self.assertEquals(data['Close'].iloc[-1], 107.06)

        # Test bad case
        with self.assertRaises(ValueError):
            get_crypto_data_for_date(CRYPTO_NAME, parse_date('2050-01-01'), WINDOW_SIZE)
