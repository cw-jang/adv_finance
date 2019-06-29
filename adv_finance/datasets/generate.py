import numpy as np
import pandas as pd
import datetime

from sklearn.datasets import make_classification
from random import gauss


def get_cls_data(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    X, cont = make_classification(n_samples=n_samples, n_features=n_features,
                                  n_informative=n_informative, n_redundant=n_redundant,
                                  random_state=0, shuffle=False)
    time_idx = pd.DatetimeIndex(periods=n_samples, freq=pd.tseries.offsets.BDay(),
                                end=pd.datetime.today())
    X = pd.DataFrame(X, index=time_idx)
    cont = pd.Series(cont, index=time_idx).to_frame('bin')
    # Create name of columns
    columns = ['I_' + str(i) for i in range(n_informative)]
    columns += ['R_' + str(i) for i in range(n_redundant)]
    columns += ['N_' + str(i) for i in range(n_features - len(columns))]
    X.columns = columns
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return X, cont



def numBusinessDays(date0, date1):
    m, date0_ = 1, date0
    while True:
        date0_ += dt.timedelta(days=1)
        if date0_ >= date1: break
        if date0_.isoweekday() < 6: m += 1
    return m


def get_sample_returns(length=1000, sigma=1, date_start=None):
    # Prepare series
    if date_start is None:
        date_start = datetime.datetime(year=2000, month=1, day=1)

    date_ = date_start

    dates = []
    while len(dates) < length:
        # Monday == 1 ... Sunday == 7
        if date_.isoweekday() < 5:
            dates.append(date_)
        date_ += datetime.timedelta(days=1)


    series = np.empty((length))
    for i in range(series.shape[0]):
        series[i] = gauss(0, sigma)
        pDay_ = datetime.date(year=dates[i].year, month=dates[i].month, day=1)

    ret = pd.DataFrame(series, index=dates)
    return ret