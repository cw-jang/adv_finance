

import pandas as pd
import numpy as np

from numpy import log, polyfit, sqrt, std, subtract


def get_daily_vol(close, lookback=100):
    """
    Snippet 3.1, page 44, Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao >> sigma_t_i, 0), and some times too low
    (tao << sigma_t_i, 0), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    :param close:
    :param lookback:
    :return:
    """

    # daily vol re-indexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))

    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 # daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0


def get_half_life(ts):
    lagged_ts = ts.shift(1).fillna(method='ffill')
    delta = ts - lagged_ts
    beta = np.polyfit(lagged_ts.dropna(), delta.dropna(), 1)[0]
    half_life = (-1 * np.log(2) / beta)

    return half_life


def get_hurst_exponent(ts):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0
