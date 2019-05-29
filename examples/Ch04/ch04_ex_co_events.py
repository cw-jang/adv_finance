import pandas as pd
import numpy as np
import datetime as dt

from adv_finance import utils, labeling, sampling


if __name__ == "__main__":
    print("main started")

    df = pd.read_csv("J:\\data\\TRADE_A233740_2019_DV.csv")
    df.timestamp = df.timestamp.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df = df.set_index('timestamp')

    df = df[['open', 'high', 'low', 'close', 'vol']].drop_duplicates()
    df = df.loc[~df.index.duplicated(keep='first')]
    close = df

    # take top N limit
    close = close[:10000]['close']

    # get triple barrier events
    daily_vol = utils.get_daily_vol(close)
    threshold = daily_vol.mean() * 0.5
    t_events = labeling.cusum_filter(close, threshold)
    v_barriers = labeling.add_vertical_barrier(t_events=t_events, close=close, num_days=1)

    pt_sl = [1, 1]
    min_ret = 0.005
    t_barrier_events = labeling.get_events(close=df['close'],
                                                t_events=t_events,
                                                pt_sl=pt_sl,
                                                target=daily_vol,
                                                min_ret=min_ret,
                                                num_threads=8,
                                                vertical_barrier_times=v_barriers,
                                                side_prediction=None)

    num_co_events = sampling.get_num_co_events(timestamps=t_barrier_events .index, t1=t_barrier_events ['t1'])

    # Sampling Weights
    out = pd.DataFrame()
    out['tW'] = sampling.get_sample_tw(t1=t_barrier_events['t1'], num_co_events=num_co_events)
    # out['w'] = sampling.get_sample_w(t1=t_barrier_events['t1'], num_co_events=num_co_events, close=close, num_threads=1)
    # out['w'] *= out.shape[0] / out['w'].sum()

    # Time decay
    tw = out['tW'].dropna()
    decay = sampling.get_time_decay(tw, last_w=.1, is_exp=False)

    print("main finished")


