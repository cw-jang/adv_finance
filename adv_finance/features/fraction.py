import numpy as np
import pandas as pd

def get_weights_ffd(d, thres, max_size=10_000):
    """
    Snippet 5.3 (page 83) The New Fixed-Width Window FracDiff Method

    :param d: int
    :param thres: float
    :param max_size: int, Set the maximum size for stability
    :return:
    """

    w = [1.]
    for k in range(1, max_size):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) <= thres:
            break

        w.append(w_)
    w = np.array(w)
    return w


def frac_diff_ffd(series, d, lag=1, thres=1e-5, max_size=10_000):
    """
    Snippet 5.3 (page 83) The New Fixed-Width Window FracDiff Method

    Compute Fractional Differentiation

    :param series:
    :param d:
    :param lag:
    :param thres:
    :param max_size:
    :return:
    """

    max_size = int(max_size / lag)
    w = get_weights_ffd(d, thres, max_size)
    width = len(w)
    series_ = series.fillna(method='ffill').dropna()
    rolling_array = []
    for i in range(width):
        rolling_array.append(series_.shift(i * lag).values)

    rolling_array = np.array(rolling_array)
    series_val = np.dot(rolling_array.T, w)
    series_val = series_val.reshape(-1, )
    timestamps = series.index[-len(series_val):]
    series = pd.Series(series_val, index=timestamps)
    return series

