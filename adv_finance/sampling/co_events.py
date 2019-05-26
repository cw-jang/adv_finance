
import pandas as pd
import numpy as np

from adv_finance.multiprocess import mp_pandas_obj


def mp_num_co_events(timestamps, t1, molecule):
    """
    Snippet 4.1 (page 60) Estimating The Uniqueness Of A Label

    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +moldecule[-1] is the date of the last event on which the weight will be computed

    Any event that starts before t1[moleucle].max() impacts the count.

    :param timestamps:
    :param t1:
    :param molecule:
    :return:
    """

    # 1) Find events that span the period [molecule[0], molecule[-1]]
    t1 = t1.fillna(timestamps[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()]    # events that start at or before t1[molecule].max()

    # 2) Count events spanning a bar
    iloc = timestamps.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=timestamps[iloc[0]:iloc[1] + 1])
    for t_in, t_out in t1.iteritems():
        count.loc[t_in:t_out] += 1

    return count.loc[molecule[0]:t1[molecule].max()]


def get_num_co_events(timestamps, t1, num_threads=1):
    """ Calculate the number of co events

    :param timestamps:
    :param t1:
    :param num_threads:
    :return:
    """

    return mp_pandas_obj(mp_num_co_events, ('molecule', t1.index),
                         num_threads, timestamps=timestamps, t1=t1)

