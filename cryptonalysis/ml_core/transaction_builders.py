import logging
import os
import random
from abc import ABCMeta, abstractmethod
from datetime import date
from market import MarketParameters
from pandas import Series

# Logging
logger = logging.getLogger()

DUMP_DIR = os.path.join('cryptonalysis', 'data', 'dump')
TRANSACTION_TYPE = {
    'BUY': 1,
    'SELL': 0,
    'UNKNOWN': -1  # Used for pure prediction
}


def get_transaction_type(transaction_value):
    """
    Get a transaction type as a string (e.g. BUY, SELL, UNKNOWN) given a transaction value using the TRANSACTION_TYPE
    constant dictionary.

    >>> get_transaction_type(0)
    'SELL'
    >>> get_transaction_type(1)
    'BUY'
    >>> get_transaction_type(-1)
    'UNKNOWN'

    :param transaction_value
    :type transaction_value: int
    :return:
    """
    for transaction, value in TRANSACTION_TYPE.items():
        if transaction_value == value:
            return transaction

    raise ValueError("Unknown transaction value {}".format(transaction_value))


class CryptoPredictor(object):
    """
    Abstract class CryptoPredictor.
    Predicts what cryptocurrency transaction to perform (buy/sell)
    given historical price data.
    The predictor can be run on a time period to estimate total ROI
    at the end based on the predicted transactions.
    """
    __metaclass__ = ABCMeta

    # Default parameters
    STARTING_INVESTMENT = 100.0  # USD
    DAILY_ALLOWANCE = 5.0
    LOOKAHEAD_DAYS = 1
    PROB_BUY = 1
    PROB_SELL = 1

    def __init__(self, market, price_list, window_size, crypto_name, starting_date, ending_date,
                 starting_investment=None, daily_allowance=None, lookahead_days=None, prob_buy=None, prob_sell=None):
        """
        CryptoPredictor constructor.
        :param market: An instance of global market parameters
        :type market: MarketParameters
        :param price_list: The list of all crypto prices to use for making transactions from the starting date
        :type price_list: Series
        :param window_size: The window size to use for making transactions
        :type window_size: int
        :param crypto_name: The cryptocurrency 3-character code (e.g., BTC, ETH, etc.)
        :type crypto_name: str
        :param starting_date: The date from which to start making transactions
        :type starting_date: date
        :param ending_date: The date in which to stop making transactions
        :type ending_date: date
        :param starting_investment: The starting investment in fiat
        :type starting_investment: float
        :param daily_allowance: The daily amount of money (in fiat) that can be invested in making transactions
        :type daily_allowance: float
        :param lookahead_days: The number of days to look ahead when making a transaction
        :type lookahead_days: int
        :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
        when it actually has to buy. This parameter is useful when estimating the ROI of a predictor knowing a model's
        accuracy.
        :type prob_buy: float
        :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
        when it actually has to sell. This parameter is useful when estimating the ROI of a predictor knowing a model's
        accuracy.
        :type prob_sell: float
        """

        # Initialize state variables
        self.total_investment = 0
        self.cash = 0
        self.owned_crypto = 0
        self.transactions = []

        self.market = market
        self._starting_date = starting_date
        self.ending_date = ending_date
        if self._starting_date >= self.ending_date or not self._starting_date or not self.ending_date:
            raise ValueError("Starting date {} must be before ending date {} and they both must be provided!".format(
                self._starting_date, self.ending_date))
        if window_size < 1:
            raise ValueError("Window size must be greater than 0.")
        self._window_size = window_size
        self._crypto_name = crypto_name

        # Price period
        self._price_list = price_list[self._starting_date:self.ending_date]

        # Override default parameters
        self.starting_investment = starting_investment or CryptoPredictor.STARTING_INVESTMENT
        self.daily_allowance = daily_allowance or CryptoPredictor.DAILY_ALLOWANCE
        self.lookahead_days = lookahead_days or CryptoPredictor.LOOKAHEAD_DAYS
        if self.lookahead_days < 1:
            raise ValueError("Lookahead days must be greater than 0.")
        if self._window_size + self.lookahead_days > len(self._price_list):
            raise ValueError("The Window size plus the Lookahead days cannot exceed the length of the price list.")

        # Probabilities (prediction accuracy)
        self.prob_buy = prob_buy or CryptoPredictor.PROB_BUY
        if self.prob_buy < 0 or self.prob_buy > 1:
            raise ValueError("prob_buy should be a decimal between 0 (inclusive) and 1 (inclusive).")

        self.prob_sell = prob_sell or CryptoPredictor.PROB_SELL
        if self.prob_sell < 0 or self.prob_sell > 1:
            raise ValueError("prob_sell should be a decimal between 0 (inclusive) and 1 (inclusive).")

    def reset_predictor_state(self):
        """
        Reset the CryptoPredictor's state variables.
        """

        self.total_investment = self.starting_investment
        self.cash = self.starting_investment
        self.owned_crypto = 0
        self.transactions = []

    def get_transaction_tuple(self, prices, current_price, future_prices):
        """
        Get a transaction (transaction, crypto_amount) based on the given parameters.
        The method makes use of the prob_buy and prob_sell properties to correctly return a transaction based on chance.
        :param prices: a list of the last self._window_size prices of the given cryptocurrency
        :param current_price: the price of the cryptocurrency at the present day
        :param future_prices: a list of future prices of the cryptocurrency of size lookahead_days
        :return: a tuple of the form (transaction, amount), where transaction can be 'BUY' or 'SELL'
        """

        # Probabilities (accuracy) for back testing
        random_draw = random.randint(0, 100)
        keep_buy = random_draw <= int(self.prob_buy * 100)
        random_draw = random.randint(0, 100)
        keep_sell = random_draw <= int(self.prob_sell * 100)

        transaction = self.get_transaction(prices, current_price, future_prices)
        if transaction == 'BUY':
            if keep_buy:
                crypto_amount, fiat_amount = self.buy(current_price)
            else:
                transaction = 'SELL'
                crypto_amount, fiat_amount = self.sell(current_price)
        elif transaction == 'SELL':
            if keep_sell:
                crypto_amount, fiat_amount = self.sell(current_price)
            else:
                transaction = 'BUY'
                crypto_amount, fiat_amount = self.buy(current_price)
        else:
            raise ValueError("get_transaction() method must return either 'BUY' or 'SELL'")

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount

    @abstractmethod
    def get_transaction(self, prices, current_price, future_prices):
        # type: (list, float, list) -> str
        """
        Child classes must override this method to define a way of doing a transaction based on the given
        parameters.
        :param prices: a list of the last self._window_size prices of the given cryptocurrency
        :param current_price: the price of the cryptocurrency at the present day
        :param future_prices: a list of future prices of the cryptocurrency of size lookahead_days
        :return: a transaction that can be 'BUY' or 'SELL'
        """
        pass

    def get_max_crypto_transaction(self, buy, current_price):
        """
        Get the maximum crypto transaction amount allowed based on transaction type, current cash,
        current price, and transaction fee.
        :param buy: True if buying, False if selling
        :param current_price: The current price (in fiat) of the cryptocurrency
        :return The max crypto amount (e.g. ETH) currently allowed on a transaction
        """
        if buy:
            # crypto_amount = \
            #     (self.cash / current_price) - ((self.cash / current_price) * self.market.transaction_fee_perc)
            # Revised formula to account for all the cash available
            crypto_amount = self.cash / (current_price + (current_price * self.market.transaction_fee_perc))
        else:
            # crypto_amount = self.owned_crypto - (self.owned_crypto * self.market.transaction_fee_perc)
            crypto_amount = self.owned_crypto  # All of it

        return crypto_amount

    def buy(self, current_price):
        """
        Simulates a BUY transaction given the current crypto price.
        The strategy is to buy what you can with your current cash. This method may be overridden if different
        strategies are to be adopted.
        :param current_price
        :return: a tuple (crypto_amount, fiat_amount)
        """
        crypto_amount = self.get_max_crypto_transaction(True, current_price)
        fiat_amount = (crypto_amount * current_price)
        fee = fiat_amount * self.market.transaction_fee_perc
        fiat_amount += fee
        return crypto_amount, fiat_amount

    def sell(self, current_price):
        """
        Simulates a SELL transaction given the current crypto price.
        The strategy is to sell all crypto. This method may be overridden if different
        strategies are to be adopted.
        :param current_price
        :return: a tuple (crypto_amount, fiat_amount)
        """
        crypto_amount = self.owned_crypto
        fiat_amount = (crypto_amount * current_price)
        fee = fiat_amount * self.market.transaction_fee_perc
        fiat_amount -= fee
        return crypto_amount, fiat_amount

    def get_transaction_fiat_amount(self, buy, crypto_amount, current_price):
        """
        Get the fiat amount of a transaction to perform.
        :param buy: True if buying, False if selling
        :param crypto_amount: The amount of crypto to buy/sell
        :param current_price: The current price (in fiat) of the cryptocurrency
        :return The fiat amount (e.g. USD) of a transaction.
        """
        if buy:
            return (crypto_amount * current_price) + (crypto_amount * current_price * self.market.transaction_fee_perc)
        else:
            return (crypto_amount * current_price) - (crypto_amount * current_price * self.market.transaction_fee_perc)

    def perform_transaction(self, buy, crypto_amount, current_price):
        """
        Perform a cryptocurrency transaction.
        :param buy: True if buying, False if selling
        :param crypto_amount: The amount of crypto to buy/sell
        :param current_price: The current price (in fiat) of the cryptocurrency
        """
        # Check for crypto transaction limits
        if crypto_amount > self.market.max_transaction_size_crypto:
            crypto_amount = self.market.max_transaction_size_crypto

        # Check that predictor is not cheating (buying/selling more than it can)
        max_crypto_amount = self.get_max_crypto_transaction(buy, current_price)
        if crypto_amount > max_crypto_amount:
            crypto_amount = max_crypto_amount

        if buy:
            fiat_amount = self.get_transaction_fiat_amount(True, crypto_amount, current_price)
            self.cash -= fiat_amount
            self.owned_crypto += crypto_amount
            assert round(self.cash, 4) >= 0
        else:
            fiat_amount = self.get_transaction_fiat_amount(False, crypto_amount, current_price)
            self.cash += fiat_amount
            self.owned_crypto -= crypto_amount
            assert self.owned_crypto >= 0

        logger.debug(
            "{0} - {1} {2} (${3})".format('BUY' if buy else 'SELL', self._crypto_name, round(crypto_amount, 4),
                                          round(fiat_amount, 2)))

    def run_predictor(self, save_roi=False):
        """
        Run the predictor, which will calculate cash and crypto amounts daily based on the transaction strategy.
        :param save_roi: Whether or not to save the ROI of this predictor run in a file
        :type save_roi: bool
        :return: a list of dictionaries with prices (training attributes) and transaction (class buy/sell)
        Example: [{[price1, price2, price3, ...], transaction: 'BUY'}, ...]
        """

        self.reset_predictor_state()

        # Start trading
        logger.info("Running predictor for {0}...".format(self.__class__.__name__))
        logger.info("Predictor parameters:\nStarting investment: ${0}, Daily allowance: ${1}, "
                    "Lookahead days: {2}, Prob buy: {3}, Prob sell: {4}".format(self.starting_investment,
                                                                                self.daily_allowance,
                                                                                self.lookahead_days,
                                                                                self.prob_buy, self.prob_sell))
        logger.info("Start date: {0}".format(self._starting_date))

        # First crypto purchase
        current_price = float(self._price_list[self._window_size - 1])
        self.perform_transaction(True, self.get_max_crypto_transaction(True, current_price), current_price)

        day = 1
        stop_day = 0
        for start_day in range(len(self._price_list) - self._window_size - (self.lookahead_days - 1)):
            stop_day = start_day + self._window_size
            logger.debug("Day {0} - {1}".format(day, self._price_list.index[stop_day]))
            time_window = self._price_list[start_day:stop_day]
            lookahead_day = stop_day + self.lookahead_days - 1
            future_prices = self._price_list[stop_day:lookahead_day + 1]
            current_price = float(self._price_list[stop_day - 1])
            transaction = self.get_transaction_tuple(time_window, current_price, future_prices)  # type: tuple

            transaction_type = transaction[0]
            crypto_amount = transaction[1]

            self.perform_transaction(transaction_type == 'BUY', crypto_amount, current_price)

            # End of the day allowance
            self.cash += self.daily_allowance
            self.total_investment += self.daily_allowance

            # Build transaction dict
            transaction_dict = {
                'prices': time_window,
                'transaction': transaction_type
            }
            self.transactions.append(transaction_dict)

            day += 1

        logger.info("Finished running predictor")
        logger.info("End date: {0}".format(self._price_list.index[stop_day]))
        self._sell_all_crypto(current_price, save_roi)  # Sell everything

        return self.transactions

    def run_transaction_simulation(self, transactions):
        """
        Run a transaction simulation given a list of transactions to perform per day.
        :param transactions: a list of transactions (as ints or as strings) to perform on the given daily prices.
        :type transactions: list of int or list of str
        """

        self.reset_predictor_state()

        logger.info("Start date: {0}".format(self._starting_date))

        # Start trading
        logger.info("Running transaction simulation...")
        logger.info("Predictor parameters:\nStarting investment: ${0}, Daily allowance: ${1}".format(
            self.starting_investment, self.daily_allowance))

        # First crypto purchase
        current_price = float(self._price_list[self._window_size - 1])
        self.perform_transaction(True, self.get_max_crypto_transaction(True, current_price), current_price)

        day = 0
        stop_day = 0
        for start_day in range(len(self._price_list) - self._window_size - (self.lookahead_days - 1)):
            if day >= len(transactions):
                break

            stop_day = start_day + self._window_size
            logger.debug("Day {0} - {1}".format(day + 1, self._price_list.index[stop_day]))
            current_price = float(self._price_list[stop_day - 1])
            t = transactions[day]
            if type(t) is int:
                transaction = get_transaction_type(t)
            elif type(t) is str:
                transaction = t
            else:
                raise TypeError('List of transactions should be either ints or strings.')

            crypto_amount, fiat_amount = self.buy(current_price) \
                if transaction == 'BUY' else self.sell(current_price)
            # Ignore for transactions less than or close to a minimum crypto/fiat transaction
            if crypto_amount < self.market.min_transaction_size_crypto or \
                    round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
                crypto_amount = 0

            self.perform_transaction(transaction == 'BUY', crypto_amount, current_price)

            # End of the day allowance
            self.cash += self.daily_allowance
            self.total_investment += self.daily_allowance

            day += 1

        logger.info("Finished running transaction simulation")
        logger.info("End date: {0}".format(self._price_list.index[stop_day]))
        self._sell_all_crypto(current_price, False)  # Sell everything

    def _sell_all_crypto(self, current_price, save_roi):
        """
        Sell all crypto in wallet given the current price.
        :param save_roi: Whether or not to save the ROI of this predictor run in a file
        :type save_roi: bool
        :param current_price: The current price (in fiat) of the crypto
        """

        logger.info("*** Before selling all crypto ***")
        logger.info("Cash: ${0}".format(round(self.cash, 2)))
        logger.info("Owned crypto: {0} {1}".format(self._crypto_name, round(self.owned_crypto, 4)))

        self.perform_transaction(False, self.get_max_crypto_transaction(False, current_price), current_price)

        logger.info("*** After selling all crypto ***")
        logger.info("Cash: ${0}".format(round(self.cash, 2)))
        logger.info("Owned crypto: {0} {1}".format(self._crypto_name, round(self.owned_crypto, 4)))
        logger.info("Total investment: ${0}".format(round(self.total_investment, 2)))
        roi = self.cash - self.total_investment
        roi_perc = roi / self.total_investment
        logger.info("ROI: ${0}".format(round(roi, 2)))
        logger.info("ROI %: {}%".format(round(roi_perc * 100, 2)))
        logger.info("*" * 20)

        # Save ROI
        if save_roi:
            if not os.path.exists(DUMP_DIR):
                os.mkdir(DUMP_DIR)
            file_name = "roi_b{}_s{}.csv".format(self.prob_buy, self.prob_sell)
            file_exists = os.path.isfile(os.path.join(DUMP_DIR, file_name))
            with open(os.path.join(DUMP_DIR, file_name), 'a') as f:
                col_names = ['crypto', 'roi', 'roi_perc', 'predictor', 'start_date', 'end_date', 'prob_buy', 'prob_sell',
                             'lookahead']
                col_vals = [self._crypto_name, round(roi, 2), round(roi_perc, 2), self.__class__.__name__,
                            self._starting_date, self.ending_date, self.prob_buy, self.prob_sell, self.lookahead_days]

                # Write headers
                if not file_exists:
                    f.write("{}\n".format(','.join(col_names)))

                line = ','.join([str(val) for val in col_vals]) + '\n'
                f.write(line)


class BiffPredictor(CryptoPredictor):
    """
    Class BiffPredictor.
    Predicts what cryptocurrency transaction to perform (buy/sell)
    given historical price data and looking at the future price after a number of "lookahead" days.
    The strategy is to buy what you can with your cash if last future price is higher than current price.
    Sell all if last future price is lower than or equal to current price.
    """

    def get_transaction(self, prices, current_price, future_prices):
        """
        Buy what you can with your cash if the last future price is higher than current price.
        Sell all if the last future price is lower than or equal to current price.
        :param prices
        :param current_price
        :param future_prices
        :return:
        """

        if future_prices[-1] > current_price:  # Buy what you can with your cash
            return 'BUY'
        else:  # Sell ALL
            return 'SELL'


class BiffPredictorSmart(CryptoPredictor):
    """
    Class BiffPredictorSmart.
    Predicts what cryptocurrency transaction to perform (buy/sell) given historical price data and looking at the future
    prices after a number of "lookahead" days.
    The strategy is to buy what you can with your cash if the current price is lower than or equal to the average future
    price.
    Sell all if the current price is greater than the average future price.
    """

    def get_transaction(self, prices, current_price, future_prices):
        """
        Buy what you can with your cash if the current price is lower than or equal to the average future
        price.
        Sell all if the current price is greater than the average future price.
        :param prices
        :param current_price
        :param future_prices
        :return:
        """

        # Strategy 1
        avg_future_price = sum(future_prices) / float(len(future_prices))

        # Strategy 2
        future_prices = list(future_prices)
        future_prices.insert(0, current_price)
        price_differences = []
        for i in range(0, len(future_prices) - 1):
            price_differences.append(future_prices[i + 1] - future_prices[i])
        s = sum(price_differences)

        if avg_future_price > current_price:
            return 'BUY'
        else:
            return 'SELL'


class ReverseBiffPredictor(CryptoPredictor):
    """
    Class ReverseBiffPredictor.
    Predicts what cryptocurrency transaction to perform (buy/sell)
    given historical price data and looking at the future prices after a number of "lookahead" days.
    The strategy is to buy what you can with your cash if last future price is lower than or equal current price.
    Sell all if last future price is higher than current price.
    """

    def get_transaction(self, prices, current_price, future_prices):
        """
        Buy what you can with your cash if last future price is lower than or equal to current price.
        Sell all if last future price is higher than current price.
        """
        if future_prices[-1] <= current_price:  # Buy what you can with your cash
            return 'BUY'
        else:  # Sell ALL
            return 'SELL'


class LazyPredictor(CryptoPredictor):
    """
    Class LazyPredictor.
    Don't do anything. Just wait to see if you get rich by lying on your sofa.
    """

    def buy(self, current_price):
        crypto_amount = 0
        fiat_amount = 0
        return crypto_amount, fiat_amount

    def sell(self, current_price):
        crypto_amount = 0
        fiat_amount = 0
        return crypto_amount, fiat_amount

    def get_transaction(self, prices, current_price, future_prices):
        """
        I'm a lazy fuck, I don't do anything...ever.
        """
        transaction = 'BUY'
        return transaction


class RandomPredictor(CryptoPredictor):
    """
    Class RandomPredictor.
    Just be random AF when making transactions.
    """

    def get_transaction(self, prices, current_price, future_prices):
        """
        Buy what you can with your cash or sell all, randomly.
        """
        transaction_no = random.randint(0, 1)
        if transaction_no == 0:
            return 'BUY'
        else:
            return 'SELL'


class GreedyPredictor(CryptoPredictor):
    """
    Class GreedyPredictor.
    Always buy with whatever cash is available. NEVER sell.
    """

    def get_transaction(self, prices, current_price, future_prices):
        """
        Buy what you can with your cash all the time.
        """
        transaction = 'BUY'
        return transaction


def get_predictor_class_from_name(predictor_cls_name):
    """
    Get a CryptoPredictor subclass given a class name.
    :param predictor_cls_name
    :type predictor_cls_name: str
    :return: A class object belonging to the required predictor class
    :rtype: type
    """
    if predictor_cls_name == 'BiffPredictor':
        return BiffPredictor
    elif predictor_cls_name == 'BiffPredictorSmart':
        return BiffPredictorSmart
    elif predictor_cls_name == 'ReverseBiffPredictor':
        return ReverseBiffPredictor
    elif predictor_cls_name == 'LazyPredictor':
        return LazyPredictor
    elif predictor_cls_name == 'RandomPredictor':
        return RandomPredictor
    elif predictor_cls_name == 'GreedyPredictor':
        return GreedyPredictor
    else:
        raise TypeError("CryptoPredictor subclass '{}' does not exist.".format(predictor_cls_name))
