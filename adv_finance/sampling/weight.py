import pandas as pd
import numpy as np

from adv_finance.multiprocess import mp_pandas_obj


def mp_sample_tw(t1, num_co_events, molecule):
    """
    Snippet 4.2 (page 62) Estimating The Average Uniqueness Of A Label

    :param timestamps: (Series): Used for assigning weight. Larger value, larger weight e.g, log return
    :param t1: (Series)
    :param num_co_events: (Series)
    :param molecule:
    :return:
    """

    # Derive average uniqueness over the event's lifespan
    weight = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[weight.index].iteritems():
        weight.loc[t_in] = (1 / num_co_events.loc[t_in:t_out]).mean()

    return weight.abs()


def get_sample_tw(t1, num_co_events, num_threads=1):
    """
    Calculate sampling weight with considering some attributes

    :param timestamps:
    :param t1:
    :param num_co_events:
    :param num_threads:
    :return:
    """

    weight = mp_pandas_obj(mp_sample_tw, ('molecule', t1.index), num_threads=num_threads,
                           t1=t1, num_co_events=num_co_events)

    return weight


def mp_sample_w(t1, num_co_events, close, molecule):
    """
    Snippet 4.10 (page 69) Determination Of Sample Weight By Absolute Return Attribution

    :param t1:
    :param num_co_events:
    :param close:
    :param molecule:
    :return:
    """

    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[wght.index].iteritems():
        wght.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()

    return wght.abs()

def get_sample_w(t1, num_co_events, close, num_threads=1):
    """
    Snippet 4.10 (page 69) Determination Of Sample Weight By Absolute Return Attribution

    :param t1:
    :param num_co_events:
    :param close:
    :param num_threads:
    :return:
    """

    wght = mp_pandas_obj(mp_sample_w, ('molecule', t1.index), num_threads=num_threads,
                       t1=t1, num_co_events=num_co_events, close=close)

    return wght