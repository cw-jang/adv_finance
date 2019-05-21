
import numpy as np
import pandas as pd

from collections import namedtuple
from numba import jit
from adv_finance.utils import ewma


@jit(nopython=True)
def numba_isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return np.fabs(a - b) <= np.fmax(rel_tol * np.fmax(np.fabs(a), np.fabs(b)), abs_tol)


@jit(nopython=True)
def get_signed_ticks_jit(t):
    bs = t
    bs[0] = 1
    for i in np.arange(1, bs.shape[0]):
        if numba_isclose(bs[i], 0.0):
            bs[i] = bs[i - 1]
    return bs


class ImbalanceBars:
    """
    2.3.2 Imbalance bars implementation
    """

    def __init__(self, metric, n_prev_bars=3, exp_n_ticks_init=1000):
        """
        Constructor

        :param df:
        :param metric:
        :param n_prev_bars:
        :param exp_n_ticks_init:
        """
        self.metric = metric
        self.n_prev_bars = n_prev_bars
        self.exp_n_ticks = exp_n_ticks_init
        self.exp_n_ticks_init = exp_n_ticks_init
        self.n_ticks_bar = [] # List of number of ticks from prev bars
        self.cache = []
        self.cache_tuple = namedtuple('CacheData', ['tm', 'price', 'high', 'low', 'cum_ticks', 'cum_vol', 'cum_theta'])


    def _get_signed_ticks(self, prices):
        """
        Applies the tick rule as defined on page 29.

        : param prices: numpy array of price
        : return: the singed tick array
        """
        return get_signed_ticks_jit(np.sign(np.diff(prices)))


    def _get_imbalance_ticks(self, df):
        signed_ticks = self._get_signed_ticks(df.PRICE.values)

        if self.metric == "tick_imbalance":
            imb_ticks = signed_ticks
        elif self.metric == "dollar_imbalance":
            imb_ticks = signed_ticks * df.DV.values[1:]
        else:
            imb_ticks = signed_ticks * df.V.values[1:]

        return imb_ticks

    def _update_cache(self, tm, price, low_price, high_price, cum_ticks, cum_vol, cum_theta):
        cache_data = self.cache_tuple(tm=tm, price=price, high=high_price, low=low_price,
                                      cum_ticks=cum_ticks, cum_vol=cum_vol, cum_theta=cum_theta)
        self.cache.append(cache_data)

    def _update_counters(self):

        if self.cache:
            latest_entry = self.cache[-1]

            cum_ticks = int(latest_entry.cum_ticks)
            cum_vol = int(latest_entry.cum_vol)
            low_price = np.float(latest_entry.low)
            high_price = np.float(latest_entry.high)
            cum_theta = np.float(latest_entry.cum_theta)
        else:
            # Reset counters
            cum_ticks, cum_theta, cum_vol = 0, 0, 0
            high_price, low_price = -np.inf, np.inf

        return cum_ticks, cum_vol, cum_theta, high_price, low_price

    def _get_expected_imbalance(self, window, imbalance_arr):

        if len(imbalance_arr) < self.exp_n_ticks_init:
            ewma_window = np.nan
        else:
            ewma_window = int(min(len(imbalance_arr), window))

        if np.isnan(ewma_window):
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                imbalance_arr[-ewma_window:], window=ewma_window)[-1]

        return expected_imbalance

    def _create_bars(self, tm, price, high_price, low_price, list_bars):
        open_price = self.cache[0].price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price
        vol = self.cache[-1].cum_vol
        start_tm = self.cache[0].tm
        
        list_bars.append([tm, open_price, high_price, low_price, close_price, vol, start_tm])


    def _extract_bars(self, imb_arr, tm_arr, data_arr):
        cum_ticks, cum_vol, cum_theta, high_price, low_price = self._update_counters()

        expected_imbalance = np.nan
        list_bars = []
        for i in np.arange(len(imb_arr)):
            tm = tm_arr[i]
            price = data_arr[i][0]
            vol = data_arr[i][1]
            imbalance = imb_arr[i]

            # update high/low_price
            if price > high_price:
                high_price = price

            if price <= low_price:
                low_price = price

            cum_ticks += 1
            cum_vol += vol
            cum_theta += imbalance

            self._update_cache(tm, price, low_price, high_price, cum_ticks, cum_vol, cum_theta)

            if not list_bars and np.isnan(expected_imbalance):
                expected_imbalance = self._get_expected_imbalance(self.exp_n_ticks, imb_arr)

            # Check expression for possible bar generation
            if np.abs(cum_theta) > self.exp_n_ticks * np.abs(expected_imbalance):
                self._create_bars(tm, price, high_price, low_price, list_bars)
                self.n_ticks_bar.append(cum_ticks)
                self.exp_n_ticks = ewma(np.array(self.n_ticks_bar[-self.n_prev_bars:], dtype=float), self.n_prev_bars)[-1]
                # n_prev_ticks = np.sum(self.n_ticks_bar[-self.n_prev_bars:])
                n_prev_bars = min(len(list_bars), self.n_prev_bars)
                expected_imbalance = self._get_expected_imbalance(self.exp_n_ticks * n_prev_bars, imb_arr)
                # expected_imbalance = self._get_expected_imbalance(self.exp_n_ticks * self.n_prev_bars, imb_arr)

                # Reset counters
                cum_ticks, cum_vol, cum_theta = 0, 0, 0
                high_price, low_price = -np.inf, np.inf

                self.cache = []
                self._update_cache(tm, price, low_price, high_price, cum_ticks, cum_vol, cum_theta)

        return list_bars

    def batch_run(self, df):
        imb_arr = self._get_imbalance_ticks(df).astype(float)
        tm_arr = df.index.values[1:]
        data_arr = df[['PRICE', 'V']].values[1:]

        imb_bars = self._extract_bars(imb_arr, tm_arr, data_arr)

        df_bars = pd.DataFrame(imb_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol', 'start'])
        return df_bars



def get_dollar_imbalance_bar(df, n_prev_bars, exp_n_ticks_init):
    bars = ImbalanceBars("dollar_imbalance", n_prev_bars, exp_n_ticks_init)
    df_bars = bars.batch_run(df)
    return df_bars