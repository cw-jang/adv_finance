import numpy as np
import pandas as pd

from adv_finance.utils.multiprocess import mp_pandas_obj


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """
    Snippet 3.2, page 45, Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (series) close prices
    :param events: (series) of indices that signify "events" (see cusum_filter function for more details)
    :param pt_sl: (array) element 0, indicates the profit taking level; element 1 is stop loss level
    :param molecule: (an array) a set of datetime indes values for processing
    :return: DataFrame of timestamps of when first barrier was touched
    """

    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events[['t1']].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index) # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index) # NaNs

    # Get events
    for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:vertical_barrier]   # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < stop_loss[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > profit_taking[loc]].index.min()  # earliest profit taking

    return out


# Snippet 3.4 page 49, Adding a Vertical Barrier
def add_vertical_barrier(t_events, close, num_days=1):
    """
    Snippet 3.4 page 49, Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    : param t_events: (series) series of events(symmetric CUSUM filter)
    : param close: (series) close prices
    : param num_days: (int) maximum number of days a trade can be active
    : return: (series) timestamps of vertical barriers
    """
    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]

    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False, side_prediction=None):
    """
    Snippet 3.6 page 50, Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (series) Close prices
    :param t_events: (series) of t_events, These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) element 0, indicates the profit taking level; element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (series) Side of the bet(long/short) as decided by the primary model
    :return: (data frame) of events
        - events.index is event's starttime
        - events['t1'] is event's endtime
        - events['trgt'] is event's target
        - events['side'] (optional) implies the algo's position side
    """

    # 1) Get target
    target = target.loc[t_events]
    target = target[target > min_ret]   # min_ret

    # 2) Get vertical barrier(max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    # 3) From events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1., index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.loc[target.index]
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
    events = events.dropna(subset=['trgt'])

    # Apply Triple Barrier
    df0 = mp_pandas_obj(func=apply_pt_sl_on_t1,
                        pd_obj=('molecule', events.index),
                        num_threads=num_threads,
                        close=close,
                        events=events,
                        pt_sl=pt_sl_)

    events['t1'] = df0.dropna(how='all').min(axis=1)    # pd.min ignores nan

    if side_prediction is None:
        events = events.drop('side', axis=1)

    return events











