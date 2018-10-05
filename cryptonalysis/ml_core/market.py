# Market (Global) parameters


class MarketParameters:
    """
    Defines global market parameters for the entire application.
    """

    # Default parameters
    TRANSACTION_FEE_PERC = 0.01
    TRANSACTION_FEE_FIAT = 0
    MIN_TRANSACTION_SIZE_CRYPTO = 0.01
    MIN_TRANSACTION_SIZE_FIAT = 1
    MAX_TRANSACTION_SIZE_CRYPTO = 100
    MAX_TRANSACTION_SIZE_FIAT = 100

    def __init__(self, transaction_fee_perc=None, transaction_fee_fiat=None,
                 min_transaction_size_crypto=None, min_transaction_size_fiat=None,
                 max_transaction_size_crypto=None, max_transaction_size_fiat=None):
        """
        MarketParameters constructor.
        :param transaction_fee_perc:
        :param transaction_fee_fiat:
        :param min_transaction_size_crypto:
        :param min_transaction_size_fiat:
        :param max_transaction_size_crypto:
        :param max_transaction_size_fiat:
        """

        if transaction_fee_perc and (transaction_fee_perc < 0 or transaction_fee_perc > 1):
            raise ValueError('Transaction fee (%) must be between 0 and 1')

        self.transaction_fee_perc = transaction_fee_perc or MarketParameters.TRANSACTION_FEE_PERC
        self.transaction_fee_fiat = transaction_fee_fiat or MarketParameters.TRANSACTION_FEE_FIAT
        self.min_transaction_size_crypto = min_transaction_size_crypto or MarketParameters.MIN_TRANSACTION_SIZE_CRYPTO
        self.min_transaction_size_fiat = min_transaction_size_fiat or MarketParameters.MIN_TRANSACTION_SIZE_FIAT
        self.max_transaction_size_crypto = max_transaction_size_crypto or MarketParameters.MAX_TRANSACTION_SIZE_CRYPTO
        self.max_transaction_size_fiat = max_transaction_size_fiat or MarketParameters.MAX_TRANSACTION_SIZE_FIAT

    def __str__(self):
        string = "TXN fee (%): {0:.2f}\n".format(self.transaction_fee_perc)
        string += "TXN fee (flat): {0}\n".format(self.transaction_fee_fiat)
        string += "Min TXN size (crypto): {0}\n".format(self.min_transaction_size_crypto)
        string += "Min TXN size (fiat): {0}\n".format(self.min_transaction_size_fiat)
        string += "Max TXN size (crypto): {0}\n".format(self.max_transaction_size_crypto)
        string += "Max TXN size (fiat): {0}".format(self.max_transaction_size_fiat)

        return string


market = MarketParameters()
