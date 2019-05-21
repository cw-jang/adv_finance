import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numba import jit
from tqdm import tqdm, tqdm_notebook

from datetime import datetime

from .imbalance_bars import get_dollar_imbalance_bar



@jit(nopython=True)
def mad_outlier(y, thresh=3.):
    '''
    compute outliers based on mad
    # args
        y: assumed to be array with shape (N,1)
        thresh: float()
    # returns
        array index of outliers
    '''
    median = np.median(y)
    diff = np.sum((y - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh


def dollar_bars(df, dv_col, n): 
    '''
    compute dollar bars
    '''
    t = df[dv_col]
    tick_n = 0
    idx = []
    for i, x in enumerate(tqdm(t)): 
        tick_n += x 
        if tick_n >= n: 
            idx.append(i)
            tick_n = 0
            continue
    
    return df.iloc[idx]



def select_sample_data(ref, sub, price_col, date): 
    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]
    return xdf, xtdf


def plot_sample_data(ref, sub, bar_type, *args, **kwds):
    f,axes=plt.subplots(3,sharex=True, sharey=True, figsize=(10,7))
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend();
    
    ref.plot(*args, **kwds, ax=axes[1], label='price', marker='o')
    sub.plot(*args, **kwds, ax=axes[2], ls='', marker='X',
             color='r', label=bar_type)

    for ax in axes[1:]: ax.legend()
    plt.tight_layout()
    
    return


def volume_bars(df, volume_col, n): 
    'compute volume bars'
    
    vols = df[volume_col]
    tick_n = 0
    idx = []
    for i, x in enumerate(tqdm(vols)): 
        tick_n += x 
        if tick_n >= n: 
            idx.append(i)
            tick_n = 0
            continue
            
    return df.iloc[idx]


def tick_bars(df, price_col, n): 
    '''
    compute tick bars 
    '''    
    prices = df[price_col]
    tick_n = 0
    idx = []
    for i, x in enumerate(tqdm(prices)): 
        tick_n += 1
        if tick_n >= n: 
            idx.append(i)
            tick_n = 0
            continue
    return df.iloc[idx]


# def agg_imbalance_bars_(df):
#     """
#     Implements the accumulation logic
#     원본: 최적화전의 구버전
#     """
#     start = df.index[0]
#     bars = []
#     for row in tqdm(df.itertuples(), position=0):
#         t_abs = row.absTheta
#         rowIdx = row.Index
#         E_bs = row.E_bs

#         t = df.loc[start:rowIdx].shape[0]
#         if t < 1: t = 1
#         if test_t_abs(t_abs, t, E_bs):
#             bars.append((start, rowIdx, t))
#             start = rowIdx

#     return bars


# @jit(nopython=True)
# def agg_imb_bars_jit(tm_arr, ts_arr, abs_theta_arr, e_bs_arr):
#     bars = []
#     start_i = 0
#     last_i = 0
#     last_tm = tm_arr[0]
#     last_ts = ts_arr[0]
#     n_tick = len(tm_arr)

#     for i in np.arange(n_tick):
#         t_abs = abs_theta_arr[i]
#         t_e_bs = e_bs_arr[i]
#         tm = tm_arr[i]

#         if tm > last_tm:
#             last_i = i
#             last_tm = tm
#             last_ts += ts_arr[i]

#         if test_t_abs(t_abs, last_ts, t_e_bs):
#             bars.append( (tm_arr[start_i], tm, last_ts) )
#             start_i = i
#             last_ts = ts_arr[i]

#     return bars

# def agg_imb_bars(df):
#     df_1 = df
#     df_1_ts = df_1.groupby(['TIME'])['E_T'].count()
#     df_1_ts = df_1_ts.rename('ts')
#     df_1_j = pd.merge(df_1, df_1_ts, left_index=True, right_index=True)

#     tm_arr = df_1_j.index.values
#     ts_arr = df_1_j['ts'].values
#     abs_theta_arr = df_1_j['absTheta'].values
#     e_bs_arr = df_1_j['E_bs'].values

#     return agg_imb_bars_jit(tm_arr, ts_arr, abs_theta_arr, e_bs_arr)