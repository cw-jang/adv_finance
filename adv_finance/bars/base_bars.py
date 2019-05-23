import numpy as np
from numba import jit


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


def get_signed_ticks(prices):
    """
    Applies the tick rule as defined on page 29.

    : param prices: numpy array of price
    : return: the singed tick array
    """
    return get_signed_ticks_jit(np.sign(np.diff(prices)))


def get_imbalance_ticks(df, metric):
    signed_ticks = get_signed_ticks(df.PRICE.values)

    if metric == "tick_imbalance" or metric == "tick_run":
        imb_ticks = signed_ticks
    elif metric == "dollar_imbalance" or metric == "dollar_run":
        imb_ticks = signed_ticks * df.DV.values[1:]
    else:
        imb_ticks = signed_ticks * df.V.values[1:]

    return imb_ticks


def update_high_low(price, high_price, low_price):
    if price > high_price:
        high_price = price

    if price <= low_price:
        low_price = price

    return high_price, low_price

def create_bars(cache, tm, price, high_price, low_price, list_bars):
    open_price = cache[0].price
    high_price = max(high_price, open_price)
    low_price = min(low_price, open_price)
    close_price = price
    vol = cache[-1].cum_vol
    start_tm = cache[0].tm

    list_bars.append([tm, open_price, high_price, low_price, close_price, vol, start_tm])