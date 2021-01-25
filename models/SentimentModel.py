import os
import json
import nltk
import numpy as np
import pandas as pd
import pypfopt as po
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from account.Account import Account


class SentimentModel():

    def __init__(self, account=Account()):
        self.account = account.get_api()
        self.count = {}
        # Buzzwords which do not actually represent tickers
        self.buzzwords = [
            'BUY', 'DD', 'ITM', 'EV', 'YOLO', 'E', 'PS', 'H',
            'OTM', 'CEO', 'USA', 'NEW', 'CFO', 'SEE',
            'UK', 'USD', 'HUGE', 'LOOK', 'PSA', 'HOLD',
            'SELL', 'REAL', 'ONE', 'MOON', 'NEED', 'EDIT',
            'RSI', 'EPS', 'ATH', 'KNOW', 'P', 'PT', 'TV', 'X'
        ]
        self.positions = {}

    def get_positions(self):
        """ Gets current positions of algo. """
        pos = self.account.list_positions()
        res = {x.symbol: int(x.qty) for x in pos}
        self.positions = res
        return res

    def get_data(self, thresh=2):
        """ Gets data. """
        date = sorted(os.listdir('scrapes'),
                      key=lambda x: pd.to_datetime(x))[-1]
        filename = sorted([x for x in os.listdir(
            f'scrapes/{date}') if x.split('.')[-1] == 'json'])[-1]
        with open(f'scrapes/{date}/{filename}', 'r') as f:
            file = json.loads(f.read())
        data = sorted(list(file.values()), key=lambda x: -x['Upvotes'])
        text = ' '.join([x['Title'] + ' ' + x['Text'] for x in data])
        # Get sentences
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)
        # Get words
        tokenizer = RegexpTokenizer(r'\w+')
        swords = stopwords.words('english')
        words = tokenizer.tokenize(text)
        words = [x for x in words if x.lower() not in swords]
        self.count = Counter(words)
        # Valid tickers
        tickers = set([x.symbol for x in self.account.list_assets()])
        # Remove buzzwords
        tickers -= set(self.buzzwords)
        kvs = sorted(((k, v)
                      for k, v in dict(self.count).items()), key=lambda x: -x[1])
        v_tickers = [(k, v) for k, v in kvs if k in tickers and v >= thresh]
        lookup = set([k for k, v in v_tickers])
        # Get scores
        sia = SentimentIntensityAnalyzer()
        scores = []
        for sentence in sentences:
            curr = {}
            tz = RegexpTokenizer(r'\w+')
            c_tickers = set(tz.tokenize(sentence)) & lookup
            if c_tickers:
                curr['tickers'] = list(c_tickers)
                curr.update(sia.polarity_scores(sentence))
                scores.append(curr)

        # Convert to pandas
        df = pd.DataFrame(scores)
        return df.drop('tickers', 1).join(df.tickers.str.join('|').str.get_dummies())

    def get_weights(self, df=None, n=2, thresh=10, long_only=True):
        """ Gets weights. """
        if df is None:
            df = self.get_data(thresh=thresh)
        unique_tickers = df.columns[4:]
        scores = {}
        for t in unique_tickers:
            scores[t] = df[df[t] == 1].mean()['compound'] * \
                (self.count[t] ** (1 / n))
        w = pd.Series(scores)
        w /= w.sum()
        if long_only:
            w = w[w > 0].dropna()
        return w.sort_values()

    def get_target_holdings(self, capital=100000, **kwargs):
        """ Gets target holdings. """
        w = self.get_weights(**kwargs)
        if w is None:
            print(f"WARNING: weights are None for {self.name}")
            return {}
        w = w[w != 0].dropna()
        prices = pd.Series(self.get_latest_prices(list(w.index)))
        # Ignore buggy prices
        w = w[prices > 0.1].dropna()
        w /= w.sum()
        prices = prices[prices > 0.1].dropna()
        sr = abs(w[w < 0].sum())
        if sr <= 0:
            sr = 1e-10
        do = po.discrete_allocation.DiscreteAllocation(
            dict(w), prices, capital * (1 + sr), sr / (1 + sr))
        res = do.greedy_portfolio()
        print(res[1])
        return res[0]

    def get_latest_prices(self, symbols):
        """ Gets midprices for list of symbols. """
        res = {}
        for ticker in symbols:
            curr_r = self.account.polygon.last_quote(ticker)
            res[ticker] = 0.5 * (float(curr_r.bidprice) + float(curr_r.askprice))
        return res

    def rebalance(self, capital=150000, order_type='limit', **kwargs):
        """ Rebalances using target weights. """
        self.account.cancel_all_orders()
        t_holdings = self.get_target_holdings(capital=capital, **kwargs)
        self.get_positions()
        # Ensure trading out of old positions
        for k in self.positions:
            if k not in t_holdings:
                t_holdings[k] = 0
        latest_prices = self.get_latest_prices(list(t_holdings.keys()))
        errors = {}
        for ticker, shares in t_holdings.items():
            curr_shares = self.positions.get(ticker, 0)
            shares -= curr_shares
            side = 'buy' if shares > 0 else 'sell'
            if shares == 0:
                continue
            try:
                if order_type == 'limit':
                    price = self.account.polygon.last_quote(ticker)
                    price = 0.5 * (float(price.bidprice) +
                                   float(price.askprice))
                    self.account.submit_order(
                        ticker, abs(shares), side, 'limit', 'day', limit_price=price
                    )
                    print(
                        f"Submitted limit order of {ticker} at ${price} for {shares} shares.")
                else:
                    self.account.submit_order(
                        ticker, abs(shares), side, 'market', 'day',
                    )
                    print(
                        f"Submitted market order of {ticker} for {shares} shares.")
            except:
                print(
                    f"Error submitting order of {ticker} for {shares} shares.")
                errors[ticker] = shares
        return f"Rebalance successful! Errors: {errors}"
