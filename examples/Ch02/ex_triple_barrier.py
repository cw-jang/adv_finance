import pandas as pd
import numpy as np

from datetime import datetime
from adv_finance import stats, labeling


if __name__ == "__main__":
    print("main started")

    df = pd.read_csv("..\\TRADE_A233740_2019_DV.csv")
    df.timestamp = df.timestamp.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df = df.set_index('timestamp')

    df = df[['open', 'high', 'low', 'close', 'vol']].drop_duplicates()
    df = df.loc[~df.index.duplicated(keep='first')]
    close = df['close']

    # daily_vol = stats.get_daily_vol(close, 20)
    # threshold = daily_vol.ewm(20).mean() * 0.5
    # side_events = labeling.cusum_filter(close, threshold)

    # ===
    daily_vol = stats.get_daily_vol(close)
    threshold = daily_vol.mean() * 0.5
    cusum_events = labeling.cusum_filter(df['close'], threshold)
    vertical_barriers = labeling.add_vertical_barrier(t_events=cusum_events, close=df['close'], num_days=1)

    pt_sl = [1, 2]
    min_ret = 0.005
    triple_barrier_events = labeling.get_events(close=df['close'],
                                                t_events=cusum_events,
                                                pt_sl=pt_sl,
                                                target=daily_vol['close'],
                                                min_ret=min_ret,
                                                num_threads=2,
                                                vertical_barrier_times=vertical_barriers,
                                                side_prediction=None)

    labels = labeling.get_bins(triple_barrier_events, df['close'])
    # print(labels.side.value_counts())

    print("main finished")
