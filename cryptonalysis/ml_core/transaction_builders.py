from random import randint
from abc import ABCMeta, abstractmethod
from datetime import date
import logging

# Logging
LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger()


class CryptoPredictor:
    """
    Abstract class CryptoPredictor.
    Predicts what cryptocurrency transaction to perform (buy/sell)
    given historical price data.
    The predictor can be run on a time period to estimate total ROI
    at the end based on the predicted transactions.
    """
    __metaclass__ = ABCMeta

    # Initial conditions
    total_investment = 0
    cash = 0
    transactions = []

    # Default parameters
    _ending_date = date.today()
    _starting_investment = 100.0  # USD
    _daily_allowance = 5.0
    _lookahead_days = 1

    def __init__(self, market, starting_date, price_list, opening_price_list, window_size, crypto_name,
                 ending_date=None, starting_investment=None, daily_allowance=None, lookahead_days=None):
        """
        CryptoPredictor constructor.
        :param market: An instance of global market parameters
        :param starting_date: The date from which to start making transactions
        :param price_list: The list of all crypto prices to use for making transactions from the starting date
        :param opening_price_list: The list of all opening crypto prices to use for making transactions from the
        starting date
        :param window_size: The window size to use for making transactions
        :param crypto_name: The cryptocurrency 3-character code (e.g., BTC, ETH, etc.)
        :param ending_date: The date in which to stop making transactions (default is today)
        :param starting_investment: The starting investment in fiat
        :param daily_allowance: The daily amount of money (in fiat) that can be invested in making transactions
        :param lookahead_days: The number of days to look ahead when making a transaction
        """

        self.market = market
        self._starting_date = starting_date
        self._ending_date = ending_date or self._ending_date
        self._window_size = window_size
        self._crypto_name = crypto_name

        # Price period
        self._price_list = price_list[starting_date:ending_date]
        self._opening_price_list = opening_price_list[starting_date:ending_date]

        # Override default parameters
        self._starting_investment = starting_investment or self._starting_investment
        self._daily_allowance = daily_allowance or self._daily_allowance
        self._lookahead_days = lookahead_days or self._lookahead_days

        # Dependent parameters
        self.total_investment += self._starting_investment
        self._starting_price = float(price_list[0])
        self.owned_crypto = self._starting_investment / self._starting_price

    @abstractmethod
    def get_transaction(self, prices, current_price, future_price):
        # type: (list, float, float) -> (str, float)
        """
        Child classes must override this method to define a way of doing a transaction based on the given
        parameters.
        :param prices: a list of the last self._window_size prices of the given cryptocurrency
        :param current_price: the price of the cryptocurrency at the present day
        :param future_price: the future price of the cryptocurrency
        :return: a tuple of the form (transaction, amount), where transaction can be 'BUY' or 'SELL'
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
            crypto_amount = (self.cash / current_price) - (self.cash / current_price) * self.market.transaction_fee_perc
        else:
            crypto_amount = self.owned_crypto - self.owned_crypto * self.market.transaction_fee_perc

        return crypto_amount

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
            assert self.cash >= 0
        else:
            fiat_amount = self.get_transaction_fiat_amount(False, crypto_amount, current_price)
            self.cash += fiat_amount
            self.owned_crypto -= crypto_amount
            assert self.owned_crypto >= 0

        logger.debug(
            "{0} - {1} {2} (${3})".format('BUY' if buy else 'SELL', self._crypto_name, crypto_amount, fiat_amount))

    def run_predictor(self):
        """
        Run the predictor, which will calculate cash and crypto amounts daily based on the
        transaction strategy.
        :return: a list of dictionaries with prices (training attributes) and transaction (class buy/sell)
        Example: [{[price1, price2, price3, ...], transaction: 'BUY'}, ...]
        """
        # Start trading
        logger.info("Running predictor for {0}...".format(self.__class__.__name__))
        logger.info("Starting date: {0}".format(self._starting_date))
        day = 1
        stop_day = 0
        current_price = 0
        for start_day in range(len(self._price_list) - self._window_size):
            stop_day = start_day + self._window_size
            logger.debug("Day {0} - {1}".format(day, self._price_list.index[stop_day]))
            time_window = self._price_list[start_day:stop_day]
            lookahead_day = stop_day + self._lookahead_days - 1
            future_price = self._price_list[lookahead_day]
            current_price = float(self._opening_price_list[stop_day])
            transaction = self.get_transaction(time_window, current_price, future_price)  # type: tuple

            if len(transaction) < 2:
                raise ValueError("get_transaction() method must return a tuple of the form (TRANSACTION_TYPE, "
                                 "CRYPTO_AMOUNT)")

            transaction_type = transaction[0]
            crypto_amount = transaction[1]

            self.perform_transaction(transaction_type == 'BUY', crypto_amount, current_price)

            # End of the day allowance
            self.cash += self._daily_allowance
            self.total_investment += self._daily_allowance

            # Build transaction dict
            transaction_dict = {
                'prices': time_window,
                'transaction': transaction_type
            }
            self.transactions.append(transaction_dict)

            day += 1

        logger.info("Finished running predictor")
        logger.info("End date: {0}".format(self._price_list.index[stop_day]))
        self._sell_all_crypto(current_price)  # Sell everything

        return self.transactions

    def _sell_all_crypto(self, current_price):
        """
        Sell all crypto in wallet given the last price.
        :param current_price: The current price (in fiat) of the crypto
        """
        self.cash += (self.owned_crypto * current_price) - (
                    self.owned_crypto * current_price * self.market.transaction_fee_perc)
        self.owned_crypto = 0

        logger.info("Cash: ${0}".format(self.cash))
        logger.info("Owned crypto: {0} {1}".format(self._crypto_name, self.owned_crypto))
        logger.info("Total investment: ${0}".format(self.total_investment))
        logger.info("ROI: ${0}".format(self.cash - self.total_investment))
        logger.info("*" * 20)


class BiffPredictor(CryptoPredictor):
    """
    Class BiffPredictor.
    Predicts what cryptocurrency transaction to perform (buy/sell)
    given historical price data and looking at the future price after a number of "lookahead" days.
    The strategy is to buy what you can with your cash if future price is higher than current price.
    Sell all if future price is lower than or equal to current price.
    """

    def get_transaction(self, prices, current_price, future_price):
        """
        Buy what you can with your cash if future price is higher than current price.
        Sell all if future price is lower than or equal to current price.
        """
        logger.debug("Cash: ${0}".format(self.cash))
        logger.debug("Owned crypto: {0} {1}".format(self._crypto_name, self.owned_crypto))
        logger.debug("Prices: {0}".format(prices))
        logger.debug("Current price: ${0}".format(current_price))
        logger.debug("Future price: ${0}".format(future_price))

        if future_price > current_price:  # Buy what you can with your cash
            transaction = 'BUY'
            crypto_amount = self.get_max_crypto_transaction(True, current_price)
            fiat_amount = (crypto_amount * current_price)
            fee = fiat_amount * self.market.transaction_fee_perc
            fiat_amount += fee
        else:  # Sell ALL
            transaction = 'SELL'
            crypto_amount = self.owned_crypto
            fiat_amount = (crypto_amount * current_price)
            fee = fiat_amount * self.market.transaction_fee_perc
            fiat_amount -= fee

        logger.debug("Transaction fee: ${0}".format(fee))
        logger.debug("Crypto amount: {0} {1}".format(self._crypto_name, crypto_amount))
        logger.debug("Fiat amount: ${0}".format(fiat_amount))

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount


class ReverseBiffPredictor(CryptoPredictor):
    """
    Class ReverseBiffPredictor.
    Predicts what cryptocurrency transaction to perform (buy/sell)
    given historical price data and looking at the future price after a number of "lookahead" days.
    The strategy is to buy what you can with your cash if future price is lower than or equal current price.
    Sell all if future price is higher than current price.
    """

    def get_transaction(self, prices, current_price, future_price):
        """
        Buy what you can with your cash if future price is lower than or equal to current price.
        Sell all if future price is higher than current price.
        """
        if future_price <= current_price:  # Buy what you can with your cash
            transaction = 'BUY'
            crypto_amount = (self.cash / current_price) - (self.cash / current_price) * self.market.transaction_fee_perc
            fiat_amount = (crypto_amount * current_price)
            fee = fiat_amount * self.market.transaction_fee_perc
            fiat_amount += fee
        else:  # Sell ALL
            transaction = 'SELL'
            crypto_amount = self.owned_crypto
            fiat_amount = (crypto_amount * current_price)
            fee = fiat_amount * self.market.transaction_fee_perc
            fiat_amount -= fee

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount


class LazyPredictor(CryptoPredictor):
    """
    Class LazyPredictor.
    Don't do anything. Just wait to see if you get rich by lying on your sofa.
    """

    def get_transaction(self, prices, current_price, future_price):
        """
        I'm a lazy fuck, I don't do anything...ever.
        """
        transaction = 'BUY'
        crypto_amount = 0
        return transaction, crypto_amount


class RandomPredictor(CryptoPredictor):
    """
    Class RandomPredictor.
    Just be random AF when making transactions.
    """

    def get_transaction(self, prices, current_price, future_price):
        """
        Buy what you can with your cash or sell all, randomly.
        """
        transaction_no = randint(0, 1)
        if transaction_no == 0:
            transaction = 'BUY'
            crypto_amount = (self.cash / current_price) - (self.cash / current_price) * self.market.transaction_fee_perc
            fiat_amount = (crypto_amount * current_price)
            fee = fiat_amount * self.market.transaction_fee_perc
            fiat_amount += fee
        else:
            transaction = 'SELL'
            crypto_amount = self.owned_crypto
            fiat_amount = (crypto_amount * current_price)
            fee = fiat_amount * self.market.transaction_fee_perc
            fiat_amount -= fee

        logger.debug("Transaction fee: ${0}".format(fee))

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount


class GreedyPredictor(CryptoPredictor):
    """
    Class GreedyPredictor.
    Always buy with whatever cash is available. NEVER sell.
    """

    def get_transaction(self, prices, current_price, future_price):
        """
        Buy what you can with your cash all the time.
        """
        transaction = 'BUY'
        crypto_amount = (self.cash / current_price) - (self.cash / current_price) * self.market.transaction_fee_perc
        fiat_amount = (crypto_amount * current_price)
        fee = fiat_amount * self.market.transaction_fee_perc
        fiat_amount += fee

        logger.debug("Transaction fee: ${0}".format(fee))

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount


class ProbabilityPredictor(CryptoPredictor):
    """
    Class ProbabilityPredictor.
    Predict 'BUY' or 'SELL' based on a probability for each transaction type.
    For instance, if prob_buy = 0.4, then prob_sell = 1 - prob_buy = 0.6.
    """

    def __init__(self, market, starting_date, price_list, opening_price_list, window_size, crypto_name,
                 ending_date=None, starting_investment=None, daily_allowance=None, lookahead_days=None, prob_buy=1,
                 prob_sell=1):
        """
        ProbabilityPredictor constructor. The parameters are the same as those in CryptoPredictor with the exception of
        prob_buy and prob_sell, which indicate the random probability [0-1] that the transaction will be 'BUY' when it
        needs to buy and 'SELL' when it needs to sell. The rules for buying or selling are the same as the
        BiffPredictor.
        :param market: An instance of global market parameters
        :param starting_date: The date from which to start making transactions
        :param price_list: The list of all crypto prices to use for making transactions from the starting date
        :param opening_price_list: The list of all opening crypto prices to use for making transactions from the
        starting date
        :param window_size: The window size to use for making transactions
        :param crypto_name: The cryptocurrency 3-character code (e.g., BTC, ETH, etc.)
        :param ending_date: The date in which to stop making transactions (default is today)
        :param starting_investment: The starting investment in fiat
        :param daily_allowance: The daily amount of money (in fiat) that can be invested in making transactions
        :param lookahead_days: The number of days to look ahead when making a transaction
        :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
        when it actually has to buy, based on BiffPredictor rules.
        :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
        when it actually has to sell, based on BiffPredictor rules.
        """
        super(ProbabilityPredictor, self).__init__(market, starting_date, price_list, opening_price_list, window_size,
                                                   crypto_name, ending_date, starting_investment, daily_allowance,
                                                   lookahead_days)

        if prob_buy < 0 or prob_buy > 1:
            raise ValueError("prob_buy should be a decimal between 0 (inclusive) and 1 (inclusive).")

        if prob_sell < 0 or prob_sell > 1:
            raise ValueError("prob_sell should be a decimal between 0 (inclusive) and 1 (inclusive).")

        self._prob_buy = prob_buy
        self._prob_sell = prob_sell

    def get_transaction(self, prices, current_price, future_price):
        """
        Get a transaction based on luck (accuracies for BUY and SELL) given prob_buy and prob_sell and BiffPredictor's
        strategy:
        Buy what you can with your cash if future price is higher than current price.
        Sell all if future price is lower than or equal to current price.
        """

        random_draw = randint(0, 100)
        if future_price > current_price:  # Buy what you can with your cash
            keep = random_draw <= int(self._prob_buy * 100)
            if keep:
                transaction = 'BUY'
                crypto_amount = self.get_max_crypto_transaction(True, current_price)
                fiat_amount = (crypto_amount * current_price)
                fee = fiat_amount * self.market.transaction_fee_perc
                fiat_amount += fee
            else:
                transaction = 'SELL'
                crypto_amount = self.owned_crypto
                fiat_amount = (crypto_amount * current_price)
                fee = fiat_amount * self.market.transaction_fee_perc
                fiat_amount -= fee
        else:  # Sell ALL
            keep = random_draw <= int(self._prob_sell * 100)
            if keep:
                transaction = 'SELL'
                crypto_amount = self.owned_crypto
                fiat_amount = (crypto_amount * current_price)
                fee = fiat_amount * self.market.transaction_fee_perc
                fiat_amount -= fee
            else:
                transaction = 'BUY'
                crypto_amount = self.get_max_crypto_transaction(True, current_price)
                fiat_amount = (crypto_amount * current_price)
                fee = fiat_amount * self.market.transaction_fee_perc
                fiat_amount += fee

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount
