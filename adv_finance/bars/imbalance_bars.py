
import numpy as np
import pandas as pd

from collections import namedtuple

from adv_finance.utils import ewma
from adv_finance.bars import base_bars


class ImbalanceBars:
    """
    2.3.2 Imbalance bars implementation
    """

    def __init__(self, metric, n_prev_bars=3, exp_n_ticks_init=1000, store_history=False):
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
        self.cache_tuple = namedtuple('CacheData', ['tm', 'price', 'high', 'low', 'cum_ticks', 'cum_vol', 'cum_theta', 'threshold'])

        self.store_history = store_history
        self.cache_history = []

    def _update_cache(self, tm, price, low_price, high_price, cum_ticks, cum_vol, cum_theta, threshold):
        cache_data = self.cache_tuple(tm=tm, price=price, high=high_price, low=low_price,
                                      cum_ticks=cum_ticks, cum_vol=cum_vol, cum_theta=cum_theta, threshold=threshold)
        self.cache.append(cache_data)
        if self.store_history:
            self.cache_history.append(cache_data)

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

            if not list_bars and np.isnan(expected_imbalance):
                expected_imbalance = self._get_expected_imbalance(self.exp_n_ticks, imb_arr)

            self._update_cache(tm, price, low_price, high_price, cum_ticks, cum_vol, cum_theta, self.exp_n_ticks * np.abs(expected_imbalance))

            # Check expression for possible bar generation
            if np.abs(cum_theta) > self.exp_n_ticks * np.abs(expected_imbalance):
                self._create_bars(tm, price, high_price, low_price, list_bars)
                self.n_ticks_bar.append(cum_ticks)
                self.exp_n_ticks = ewma(np.array(self.n_ticks_bar[-self.n_prev_bars:], dtype=float), self.n_prev_bars)[-1]
                expected_imbalance = self._get_expected_imbalance(self.exp_n_ticks * n_prev_bars, imb_arr)

                # Reset counters
                cum_ticks, cum_vol, cum_theta = 0, 0, 0
                high_price, low_price = -np.inf, np.inf

                self.cache = []
                self._update_cache(tm, price, low_price, high_price, cum_ticks, cum_vol, cum_theta, self.exp_n_ticks * np.abs(expected_imbalance))

        return list_bars


    def batch_run(self, df):
        imb_arr = base_bars.get_imbalance_ticks(df, self.metric).astype(float)
        tm_arr = df.index.values[1:]
        data_arr = df[['PRICE', 'V']].values[1:]

        imb_bars = self._extract_bars(imb_arr, tm_arr, data_arr)
        df_bars = pd.DataFrame(imb_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol', 'start'])
        return df_bars



def get_dollar_imbalance_bars(df, n_prev_bars, exp_n_ticks_init, store_history=False):
    bars = ImbalanceBars("dollar_imbalance", n_prev_bars, exp_n_ticks_init, store_history)
    df_bars = bars.batch_run(df)

    if store_history:
        df_history = pd.DataFrame(bars.cache_history, columns=['tm', 'price', 'high', 'low', 'cum_ticks', 'cum_vol', 'cum_theta', 'threshold'])
        return df_bars, df_history

    return df_bars
