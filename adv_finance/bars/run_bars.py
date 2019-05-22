import numpy as np
import pandas as pd

from collections import namedtuple
from adv_finance.utils import ewma
from adv_finance.bars import base_bars


class RunBars:
    def __init__(self, metric, n_prev_bars, exp_n_ticks_init, store_history=False):

        self.metric = metric
        self.exp_n_ticks_init = exp_n_ticks_init
        self.exp_n_ticks = self.exp_n_ticks_init
        self.n_prev_bars = n_prev_bars
        self.n_ticks_bar = []

        self.cache = []
        self.cache_tuple = namedtuple('CacheData',
                                      ['tm', 'price', 'high', 'low', 'cum_ticks', 'cum_vol',
                                       'cum_theta_buy', 'cum_theta_sell', 'threshold'])

        self.exp_buy_proportion, self.exp_sell_proportion = np.nan, np.nan

        self.store_history = store_history
        self.cache_history = []


    def _update_counters(self):
        if self.cache:
            latest_entry = self.cache[-1]

            cum_ticks = int(latest_entry.cum_ticks)
            cum_vol = int(latest_entry.cum_vol)
            low_price = np.float(latest_entry.low)
            high_price = np.float(latest_entry.high)
            cum_theta_buy = np.float(latest_entry.cum_theta_buy)
            cum_theta_sell = np.float(latest_entry.cum_theta_sell)

        else:
            # Reset counters
            cum_ticks, cum_vol, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf

        return cum_ticks, cum_vol, cum_theta_buy, cum_theta_sell, high_price, low_price


    def _update_cache(self, tm, price, low_price, high_price, cum_theta_buy, cum_theta_sell, cum_ticks, cum_vol, threshold):

        cache_data = self.cache_tuple(tm=tm, price=price, high=high_price, low=low_price,
                                      cum_ticks=cum_ticks, cum_vol=cum_vol, cum_theta_buy=cum_theta_buy, cum_theta_sell=cum_theta_sell, threshold=threshold)

        self.cache.append(cache_data)

        if self.store_history:
            self.cache_history.append(cache_data)


    def _get_expected_imbalance(self, window, imbalance_arr):

        if len(imbalance_arr['buy']) < self.exp_n_ticks_init:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            ewma_window = int(min(len(imbalance_arr), window))

        if np.isnan(ewma_window):
            exp_buy_proportion, exp_sell_proportion = np.nan, np.nan
        else:
            buy_sample = np.array(imbalance_arr['buy'][-ewma_window:], dtype=float)
            sell_sample = np.array(imbalance_arr['sell'][-ewma_window:], dtype=float)
            buy_and_sell_imb = sum(buy_sample) + sum(sell_sample)
            exp_buy_proportion = ewma(buy_sample, window=ewma_window)[-1] / buy_and_sell_imb
            exp_sell_proportion = ewma(sell_sample, window=ewma_window)[-1] / buy_and_sell_imb

        return exp_buy_proportion, exp_sell_proportion


    def _extract_bars(self, imb_arr, tm_arr, data_arr):

        cum_ticks, cum_vol, cum_theta_buy, cum_theta_sell, high_price, low_price = self._update_counters()
        list_bars = []
        imbalance_arr = {'buy': [], 'sell': []}

        for i in np.arange(len(imb_arr)):
            tm = tm_arr[i]
            price = data_arr[i][0]
            vol = data_arr[i][1]
            imbalance = imb_arr[i]
            cum_ticks += 1
            cum_vol += vol

            # update high/low_price
            if price > high_price:
                high_price = price

            if price <= low_price:
                low_price = price

            if imbalance > 0:
                imbalance_arr['buy'].append(imbalance)
                imbalance_arr['sell'].append(0.0)
                cum_theta_buy += imbalance
            elif imbalance < 0:
                imbalance_arr['buy'].append(0.0)
                imbalance_arr['sell'].append(abs(imbalance))
                cum_theta_sell += abs(imbalance)

            if not list_bars and np.isnan(self.exp_buy_proportion):
                self.exp_buy_proportion, self.exp_sell_proportion = self._get_expected_imbalance(
                    self.exp_n_ticks, imbalance_arr)

            # Check expression for possible bar generation
            max_proportion = max(self.exp_buy_proportion, self.exp_sell_proportion)
            self._update_cache(tm, price, low_price, high_price, cum_theta_buy, cum_theta_sell, cum_ticks, cum_vol,
                               self.exp_n_ticks * max_proportion)

            if max(cum_theta_buy, cum_theta_sell) > self.exp_n_ticks * max_proportion:
                base_bars.create_bars(self.cache, tm, price, high_price, low_price, list_bars)
                self.n_ticks_bar.append(cum_ticks)
                self.exp_n_ticks = ewma(np.array(self.n_ticks_bar[-self.n_prev_bars:], dtype=float), self.n_prev_bars)[-1]
                self.exp_buy_proportion, self.exp_sell_proportion = self._get_expected_imbalance(self.exp_n_ticks * self.n_prev_bars,
                                                                                                 imbalance_arr)

                # Reset counters
                cum_ticks, cum_vol, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0
                high_price, low_price = -np.inf, np.inf
                self.cache = []

                max_proportion = max(self.exp_buy_proportion, self.exp_sell_proportion)
                self._update_cache(tm, price, low_price, high_price, cum_theta_buy, cum_theta_sell, cum_ticks, cum_vol,
                                   self.exp_n_ticks * max_proportion)

        return list_bars


    def batch_run(self, df):
        imb_arr = base_bars.get_imbalance_ticks(df, self.metric).astype(float)
        tm_arr = df.index.values[1:]
        data_arr = df[['PRICE', 'V']].values[1:]

        run_bars = self._extract_bars(imb_arr, tm_arr, data_arr)
        df_bars = pd.DataFrame(run_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol', 'start'])
        return df_bars


def get_dollar_run_bars(df, n_prev_bars, exp_n_ticks_init, store_history):

    bars = RunBars(metric='dollar_run', n_prev_bars=n_prev_bars, exp_n_ticks_init=exp_n_ticks_init, store_history=store_history)
    df_bars = bars.batch_run(df)

    if store_history:
        df_history = pd.DataFrame(bars.cache_history, columns=['tm', 'price', 'high', 'low', 'cum_ticks', 'cum_vol',
                                                               'cum_theta_buy', 'cum_theta_sell', 'threshold'])
        return df_bars, df_history

    return df_bars