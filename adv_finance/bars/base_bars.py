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

    if metric == "tick_imbalance":
        imb_ticks = signed_ticks
    elif metric == "dollar_imbalance":
        imb_ticks = signed_ticks * df.DV.values[1:]
    else:
        imb_ticks = signed_ticks * df.V.values[1:]

    return imb_ticks
