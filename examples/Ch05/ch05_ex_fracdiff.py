import pandas as pd
import numpy as np
import datetime as dt

from adv_finance import features


def getWeights(d, size):
    # thres > 0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)

    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff_(series, d, thres=.01):
    """
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])

    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]

    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()

        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]

            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs

            df_[loc] = np.dot(w[-(iloc + 1):, :].T,
                              seriesF.loc[:loc])[0, 0]

        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


if __name__ == "__main__":
    print(__file__)

    df = pd.read_csv("J:\\data\\TRADE_A233740_2019.csv")
    df.TIMESTAMP = df.TIMESTAMP.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df = df.set_index('TIMESTAMP')

    # df = df[['open', 'high', 'low', 'close', 'vol']].drop_duplicates()
    df = df.loc[~df.index.duplicated(keep='first')]
    close = df[['PRICE']]
    close.columns = ['close']

    # take top N limit
    close0 = close.loc['2019-03-04']
    # std_df = fracDiff(close0, 0.9, thres=1e-2)

    frac_df = features.frac_diff_ffd(close0['close'], 0.9, thres=.01)
    # frac_df = features.frac_diff_FFD(close0, 0.9, thres=.01)

    print('Finished')


