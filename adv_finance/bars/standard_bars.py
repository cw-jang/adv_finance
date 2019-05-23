
import numpy as np
import pandas as pd

from collections import namedtuple
from adv_finance.bars import base_bars


class StandardBars:

    def __init__(self, metric, threshold=50_000):
        self.metric = metric
        self.threshold = threshold

        self.cache = []
        self.cache_tuple = namedtuple('CacheData',
                                      ['tm', 'price', 'high', 'low', 'cum_ticks', 'cum_vol', 'cum_dollar'])

    def _update_counters(self):

        if self.cache:
            latest_entry = self.cache[-1]

            cum_ticks = int(latest_entry.cum_ticks)
            cum_dollar_value = np.float(latest_entry.cum_dollar)
            cum_vol = latest_entry.cum_vol
            low_price = np.float(latest_entry.low)
            high_price = np.float(latest_entry.high)
        else:
            # Reset counters
            cum_ticks, cum_dollar_value, cum_vol, high_price, low_price = 0, 0, 0, -np.inf, np.inf

        return cum_ticks, cum_dollar_value, cum_vol, high_price, low_price


    def _update_cache(self, tm, price, low_price, high_price, cum_ticks, cum_vol, cum_dollar_value):

        cache_data = self.cache_tuple(tm, price, high_price, low_price, cum_ticks, cum_vol, cum_dollar_value)
        self.cache.append(cache_data)


    def _extract_bars(self, tm_arr, data_arr):

        cum_ticks, cum_dollar_value, cum_vol, high_price, low_price = self._update_counters()

        list_bars = []
        for i in np.arange(len(tm_arr)):
            tm = tm_arr[i]
            price = data_arr[i][0]
            vol = data_arr[i][1]
            dollar_value = data_arr[i][2]

            high_price, low_price = base_bars.update_high_low(price, high_price, low_price)

            cum_ticks += 1
            cum_dollar_value += dollar_value
            cum_vol += vol

            self._update_cache(tm, price, low_price, high_price, cum_ticks, cum_vol, cum_dollar_value)

            # If threshold reached then take a sample
            if eval(self.metric) >= self.threshold:
                base_bars.create_bars(self.cache, tm, price, high_price, low_price, list_bars)

                # Reset counters
                cum_ticks, cum_dollar_value, cum_vol, high_price, low_price = 0, 0, 0, -np.inf, np.inf
                self.cache = []
                self._update_cache(tm, price, low_price, high_price, cum_ticks, cum_vol, cum_dollar_value)

        return list_bars


    def batch_run(self, df):
        tm_arr = df.index.values[1:]
        data_arr = df[['PRICE', 'V', 'DV']].values[1:]
        bars = self._extract_bars(tm_arr, data_arr)
        df_bars = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol', 'start'])
        return df_bars


def get_dollar_bars(df, threshold=1000_000_000):
    # bars = ImbalanceBars("dollar_imbalance", n_prev_bars, exp_n_ticks_init, store_history)
    bars = StandardBars(metric="cum_dollar_value", threshold=threshold)
    df_bars = bars.batch_run(df)
    return df_bars