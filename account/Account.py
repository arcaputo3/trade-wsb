import os

import alpaca_trade_api as tradeapi


class Account:
    """ A class to simplify access to accounts. """

    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv(f'ALPACA_KEY'),
            os.getenv(f'ALPACA_SECRET_KEY'),
            os.getenv(f'ALPACA_URL'),
            api_version='v2'
        )

    def get_api(self):
        """ Returns alpaca api. """
        return self.api
