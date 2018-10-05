import random
from abc import ABCMeta, abstractmethod
from datetime import date
import logging
import config

# Logging
logging.basicConfig(level=config.LOGGING_LEVEL)
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

    # Default parameters
    ENDING_DATE = date.today()
    STARTING_INVESTMENT = 100.0  # USD
    DAILY_ALLOWANCE = 5.0
    LOOKAHEAD_DAYS = 1

    def __init__(self, market, starting_date, price_list, window_size, crypto_name, ending_date=None,
                 starting_investment=None, daily_allowance=None, lookahead_days=None, prob_buy=1, prob_sell=1):
        """
        CryptoPredictor constructor.
        :param market: An instance of global market parameters
        :param starting_date: The date from which to start making transactions
        :param price_list: The list of all crypto prices to use for making transactions from the starting date
        :param window_size: The window size to use for making transactions
        :param crypto_name: The cryptocurrency 3-character code (e.g., BTC, ETH, etc.)
        :param ending_date: The date in which to stop making transactions (default is today)
        :param starting_investment: The starting investment in fiat
        :param daily_allowance: The daily amount of money (in fiat) that can be invested in making transactions
        :param lookahead_days: The number of days to look ahead when making a transaction
        :param prob_buy: A value between 0 and 1 which indicates the probability that the transaction will be 'BUY'
        when it actually has to buy. This parameter is useful when estimating the ROI of a predictor knowing a model's
        accuracy.
        :param prob_sell: A value between 0 and 1 which indicates the probability that the transaction will be 'SELL'
        when it actually has to sell. This parameter is useful when estimating the ROI of a predictor knowing a model's
        accuracy.
        """

        # Initial conditions
        self.total_investment = 0
        self.cash = 0
        self.owned_crypto = 0
        self.transactions = []

        self.market = market
        self._starting_date = starting_date
        self.ending_date = ending_date or CryptoPredictor.ENDING_DATE
        self._window_size = window_size
        self._crypto_name = crypto_name

        # Price period
        self._price_list = price_list[self._starting_date:self.ending_date]

        # Override default parameters
        self.starting_investment = starting_investment or CryptoPredictor.STARTING_INVESTMENT
        self.daily_allowance = daily_allowance or CryptoPredictor.DAILY_ALLOWANCE
        self.lookahead_days = lookahead_days or CryptoPredictor.LOOKAHEAD_DAYS

        # Probabilities (prediction accuracy)
        if prob_buy < 0 or prob_buy > 1:
            raise ValueError("prob_buy should be a decimal between 0 (inclusive) and 1 (inclusive).")
        self.prob_buy = prob_buy

        if prob_sell < 0 or prob_sell > 1:
            raise ValueError("prob_sell should be a decimal between 0 (inclusive) and 1 (inclusive).")
        self.prob_sell = prob_sell

    @abstractmethod
    def get_transaction(self, prices, current_price, future_prices):
        # type: (list, float, list) -> (str, float)
        """
        Child classes must override this method to define a way of doing a transaction based on the given
        parameters. The method may also make use of the prob_buy and prob_sell properties to correctly return a
        transaction based on chance.
        :param prices: a list of the last self._window_size prices of the given cryptocurrency
        :param current_price: the price of the cryptocurrency at the present day
        :param future_prices: a list of future prices of the cryptocurrency of size lookahead_days
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
            "{0} - {1} {2} (${3})".format('BUY' if buy else 'SELL', self._crypto_name, round(crypto_amount, 4),
                                          round(fiat_amount, 2)))

    def run_predictor(self):
        """
        Run the predictor, which will calculate cash and crypto amounts daily based on the transaction strategy.
        :return: a list of dictionaries with prices (training attributes) and transaction (class buy/sell)
        Example: [{[price1, price2, price3, ...], transaction: 'BUY'}, ...]
        """

        # Initial conditions
        self.total_investment = self.starting_investment
        self.cash = self.starting_investment
        self.owned_crypto = 0
        self.transactions = []

        # Start trading
        logger.info("Running predictor for {0}...".format(self.__class__.__name__))
        logger.info("Predictor parameters:\nStarting investment: ${0}, Daily allowance: ${1}, "
                    "Lookahead days: {2}".format(self.starting_investment, self.daily_allowance, self.lookahead_days))
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
            transaction = self.get_transaction(time_window, current_price, future_prices)  # type: tuple

            if len(transaction) < 2:
                raise ValueError("get_transaction() method must return a tuple of the form (TRANSACTION_TYPE, " +
                                 "CRYPTO_AMOUNT)")

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
        self._sell_all_crypto(current_price)  # Sell everything

        return self.transactions

    def _sell_all_crypto(self, current_price):
        """
        Sell all crypto in wallet given the current price.
        :param current_price: The current price (in fiat) of the crypto
        """

        logger.info("*** Before selling all crypto ***")
        logger.info("Cash: ${0}".format(round(self.cash, 2)))
        logger.info("Owned crypto: {0} {1}".format(self._crypto_name, round(self.owned_crypto, 4)))

        self.cash += (self.owned_crypto * current_price) - (
                    self.owned_crypto * current_price * self.market.transaction_fee_perc)
        self.owned_crypto = 0

        logger.info("*** After selling all crypto ***")
        logger.info("Cash: ${0}".format(round(self.cash, 2)))
        logger.info("Owned crypto: {0} {1}".format(self._crypto_name, round(self.owned_crypto, 4)))
        logger.info("Total investment: ${0}".format(round(self.total_investment, 2)))
        roi = self.cash - self.total_investment
        logger.info("ROI: ${0}".format(round(roi, 2)))
        logger.info("*" * 20)

        # Save ROI
        with open('roi.txt', 'a') as f:
            f.write(str(roi) + '\n')


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
        """
        logger.debug("Cash: ${0}".format(round(self.cash, 2)))
        logger.debug("Owned crypto: {0} {1}".format(self._crypto_name, round(self.owned_crypto, 4)))
        logger.log(logging.NOTSET, "Prices: {0}".format(prices))
        logger.debug("Current price: ${0}".format(current_price))
        logger.debug("Future price: ${0}".format(future_prices[-1]))

        if future_prices[-1] > current_price:  # Buy what you can with your cash
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

        logger.debug("Transaction fee: ${0}".format(round(fee, 2)))
        logger.debug("Crypto amount: {0} {1}".format(self._crypto_name, round(crypto_amount, 4)))
        logger.debug("Fiat amount: ${0}".format(round(fiat_amount, 2)))

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount


class BiffPredictorSmart(CryptoPredictor):
    """
    Class BiffPredictorSmart.
    Predicts what cryptocurrency transaction to perform (buy/sell) given historical price data and looking at the future
    prices after a number of "lookahead" days.
    The strategy is to buy what you can with your cash if the future price is higher than current price.
    Sell all if future price is lower than or equal to current price.
    """

    # Buy what you can with your cash
    def buy(self, current_price):
        transaction = 'BUY'
        crypto_amount = self.get_max_crypto_transaction(True, current_price)
        fiat_amount = (crypto_amount * current_price)
        fee = fiat_amount * self.market.transaction_fee_perc
        fiat_amount += fee
        return transaction, crypto_amount, fiat_amount

    # Sell ALL
    def sell(self, current_price):
        transaction = 'SELL'
        crypto_amount = self.owned_crypto
        fiat_amount = (crypto_amount * current_price)
        fee = fiat_amount * self.market.transaction_fee_perc
        fiat_amount -= fee
        return transaction, crypto_amount, fiat_amount

    def get_transaction(self, prices, current_price, future_prices):
        """
        Buy what you can with your cash if last future price is higher than current price.
        Sell all if last future price is lower than or equal to current price.
        """

        avg_future_price = sum(future_prices) / len(future_prices)  # Strategy 1

        future_prices = list(future_prices)  # Strategy 2
        future_prices.insert(0, current_price)
        price_differences = []
        for i in range(0, len(future_prices) - 1):
            price_differences.append(future_prices[i + 1] - future_prices[i])
        s = sum(price_differences)

        random_draw = random.randint(0, 100)
        keep_buy = random_draw <= int(self.prob_buy * 100)
        random_draw = random.randint(0, 100)
        keep_sell = random_draw <= int(self.prob_sell * 100)
        if current_price <= avg_future_price:
            transaction, crypto_amount, fiat_amount = self.buy(current_price) if keep_buy else self.sell(current_price)
        else:
            transaction, crypto_amount, fiat_amount = self.sell(current_price) if keep_sell else self.buy(current_price)

        # Ignore for transactions less than or close to a minimum crypto/fiat transaction
        if crypto_amount < self.market.min_transaction_size_crypto or \
                round(fiat_amount, 0) < self.market.min_transaction_size_fiat:
            crypto_amount = 0

        return transaction, crypto_amount


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

    def get_transaction(self, prices, current_price, future_prices):
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

    def get_transaction(self, prices, current_price, future_prices):
        """
        Buy what you can with your cash or sell all, randomly.
        """
        transaction_no = random.randint(0, 1)
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

    def get_transaction(self, prices, current_price, future_prices):
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
    For instance, if prob_buy = 0.4 and prob_sell = 0.7, then the predictor has a 40% chance of correctly guessing when
    to buy, and a 70% chance of correctly guessing when to sell, using the BiffPredictor buy/sell strategy.
    """

    def __init__(self, market, starting_date, price_list, window_size, crypto_name, ending_date=None,
                 starting_investment=None, daily_allowance=None, lookahead_days=None, prob_buy=1, prob_sell=1):
        """
        ProbabilityPredictor constructor. The parameters are the same as those in CryptoPredictor with the exception of
        prob_buy and prob_sell, which indicate the random probability [0-1] that the transaction will be 'BUY' when it
        needs to buy and 'SELL' when it needs to sell. The rules for buying or selling are the same as the
        BiffPredictor.
        :param market: An instance of global market parameters
        :param starting_date: The date from which to start making transactions
        :param price_list: The list of all crypto prices to use for making transactions from the starting date
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
        super(ProbabilityPredictor, self).__init__(market, starting_date, price_list, window_size, crypto_name,
                                                   ending_date, starting_investment, daily_allowance, lookahead_days)

        if prob_buy < 0 or prob_buy > 1:
            raise ValueError("prob_buy should be a decimal between 0 (inclusive) and 1 (inclusive).")

        if prob_sell < 0 or prob_sell > 1:
            raise ValueError("prob_sell should be a decimal between 0 (inclusive) and 1 (inclusive).")

        self._prob_buy = prob_buy
        self._prob_sell = prob_sell

    def get_transaction(self, prices, current_price, future_prices):
        """
        Get a transaction based on luck (accuracies for BUY and SELL) given prob_buy and prob_sell and BiffPredictor's
        strategy:
        Buy what you can with your cash if last future price is higher than current price.
        Sell all if last future price is lower than or equal to current price.
        """

        random_draw = random.randint(0, 100)
        if future_prices[-1] > current_price:  # Buy what you can with your cash
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
