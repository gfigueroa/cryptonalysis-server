# Market (Global) parameters


class MarketParameters:
    """
    Defines global market parameters for the entire application.
    """

    transaction_fee_perc = 0.01
    transaction_fee_flat = 0
    min_transaction_size_crypto = 0.1
    min_transaction_size_fiat = 1
    max_transaction_size_crypto = 100
    max_transaction_size_fiat = 100

    def __init__(self, transaction_fee_perc=None, transaction_fee_flat=None,
                 min_transaction_size_crypto=None, min_transaction_size_fiat=None,
                 max_transaction_size_crypto=None, max_transaction_size_fiat=None):
        """

        :param transaction_fee_perc:
        :param transaction_fee_flat:
        :param min_transaction_size_crypto:
        :param min_transaction_size_fiat:
        :param max_transaction_size_crypto:
        :param max_transaction_size_fiat:
        """

        if transaction_fee_perc and (transaction_fee_perc < 0 or transaction_fee_perc > 1):
            raise ValueError('Transaction fee (%) must be between 0 and 1')

        self.transaction_fee_perc = transaction_fee_perc or self.transaction_fee_perc
        self.transaction_fee_flat = transaction_fee_flat or self.transaction_fee_flat
        self.min_transaction_size_crypto = min_transaction_size_crypto or self.min_transaction_size_crypto
        self.min_transaction_size_fiat = min_transaction_size_fiat or self.min_transaction_size_fiat
        self.max_transaction_size_crypto = max_transaction_size_crypto or self.max_transaction_size_crypto
        self.max_transaction_size_fiat = max_transaction_size_fiat or self.max_transaction_size_fiat

    def __str__(self):
        string = "TXN fee (%): {0:.2f}\n".format(self.transaction_fee_perc)
        string += "TXN fee (flat): {0}\n".format(self.transaction_fee_flat)
        string += "Min TXN size (crypto): {0}\n".format(self.min_transaction_size_crypto)
        string += "Min TXN size (fiat): {0}\n".format(self.min_transaction_size_fiat)
        string += "Max TXN size (crypto): {0}\n".format(self.max_transaction_size_crypto)
        string += "Max TXN size (fiat): {0}".format(self.max_transaction_size_fiat)

        return string


market = MarketParameters()
