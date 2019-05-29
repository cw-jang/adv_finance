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


def get_time_decay(tw, last_w=1., truncate=0, is_exp=False):
    """
    Snippet 4.11 (page 70) Implementation Of Time-Decay Factors

    :param tw:
    :param last_w:
    :param truncate:
    :param is_exp:
    :return:
    """

    cum_w = tw.sort_index().cumsum()
    init_w = 1.

    if is_exp:
        init_w = np.log(init_w)

    if last_w >= 0:
        if is_exp:
            last_w = np.log(last_w)
        slope = (init_w - last_w) / cum_w.iloc[-1]
    else:
        slope = init_w / ((last_w + 1) * cum_w.iloc[-1])

    const = init_w - slope * cum_w.iloc[-1]
    weights = const + slope * cum_w

    if is_exp:
        weights = np.exp(weights)

    weights[weights < truncate]  = 0
    return weights
