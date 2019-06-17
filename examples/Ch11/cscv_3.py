# http://quantresearch.org/CSCV_3.py.txt

# !/usr/bin/env python
# On 20130704 by lopezdeprado@lbl.gov
import numpy as np, scipy.stats as ss, pandas as pd, datetime as dt
from random import gauss
from itertools import product


# ----------------------------------------------------------------------------------------
def getRefDates_MonthBusinessDate(dates):
    refDates, pDay = {}, []
    first = dt.date(year=dates[0].year, month=dates[0].month, day=1)
    m = dates[0].month
    d = numBusinessDays(first, dates[0]) + 1
    for i in dates:
        if m != i.month: m, d = i.month, 1
        pDay.append(d)
        d += 1
    for j in range(1, 30):
        lst = [dates[i] for i in range(len(dates)) if pDay[i] == j]
        refDates[j] = lst
    return refDates


# ----------------------------------------------------------------------------------------
def numBusinessDays(date0, date1):
    m, date0_ = 1, date0
    while True:
        date0_ += dt.timedelta(days=1)
        if date0_ >= date1: break
        if date0_.isoweekday() < 6: m += 1
    return m


# ----------------------------------------------------------------------------------------
def getTrades(series, dates, refDates, exit, stopLoss, side):
    # Get trades
    trades, pnl, position_, j, num = [], 0, 0, 0, None
    for i in range(1, len(dates)):
        # Find next roll and how many trading dates to it
        if dates[i] >= refDates[j]:
            if dates[i - 1] < refDates[j]: num, pnl = 0, 0
            if j < len(refDates) - 1:
                while dates[i] > refDates[j]: j += 1
        if num == None: continue
        # Trading rule
        position = 0
        if num < exit and pnl > stopLoss: position = side
        if position != 0 or position_ != 0:
            trades.append([dates[i], num, position, position_ * (series[i] - series[i - 1])])
            pnl += trades[-1][3]
            position_ = position
        num += 1
    return trades


# ----------------------------------------------------------------------------------------
def computePSR(stats, obs, sr_ref=0, moments=4):
    # Compute PSR
    stats_ = [0, 0, 0, 3]
    stats_[:moments] = stats[:moments]
    sr = stats_[0] / stats_[1]
    psrStat = (sr - sr_ref) * (obs - 1) ** 0.5 / (1 - sr * stats_[2] + sr ** 2 * (stats_[3] - 1) / 4.) ** 0.5
    psr = ss.norm.cdf((sr - sr_ref) * (obs - 1) ** 0.5 / (1 - sr * stats_[2] + sr ** 2 * (stats_[3] - 1) / 4.) ** 0.5)
    return psrStat, psr


# ----------------------------------------------------------------------------------------
def attachTimeSeries(series, series_, index=None, label='', how='outer'):
    # Attach a time series to a pandas dataframe
    if not isinstance(series_, pd.DataFrame):
        series_ = pd.DataFrame({label: series_}, index=index)
    elif label != '':
        series_.columns = [label]
    if isinstance(series, pd.DataFrame):
        series = series.join(series_, how=how)
    else:
        series = series_.copy(deep=True)
    return series


# ----------------------------------------------------------------------------------------
def evalPerf(pnl, date0, date1, sr_ref=0):
    freq = float(len(pnl)) / ((date1 - date0).days + 1) * 365.25
    m1 = np.mean(pnl)
    m2 = np.std(pnl)
    m3 = ss.skew(pnl)
    m4 = ss.kurtosis(pnl, fisher=False)
    sr = m1 / m2 * freq ** .5
    psr = computePSR([m1, m2, m3, m4], len(pnl), sr_ref=sr_ref / freq ** .5, moments=4)[0]
    return sr, psr, freq


# ----------------------------------------------------------------------------------------
def backTest(nDays=0, factor=0):
    # 1) Input parameters --- to be changed by the user
    holdingPeriod, sigma, stopLoss, length = 20, 1, 10, 1000
    # 2) Prepare series
    date_, dates = dt.date(year=2000, month=1, day=1), []
    while len(dates) < length:
        if date_.isoweekday() < 5: dates.append(date_)
        date_ += dt.timedelta(days=1)
    series = np.empty((length))
    for i in range(series.shape[0]):
        series[i] = gauss(0, sigma)
        pDay_ = dt.date(year=dates[i].year, month=dates[i].month, day=1)
        if numBusinessDays(pDay_, dates[i]) <= nDays:
            series[i] += sigma * factor
    series = np.cumsum(series)
    # 3) Optimize
    refDates = getRefDates_MonthBusinessDate(dates)
    psr, sr, trades, sl, freq, pDay, pnl, count = None, None, None, None, None, None, None, 0
    for pDay_ in refDates.keys():
        refDates_ = refDates[pDay_]
        if len(refDates_) == 0: continue
        # 4) Get trades
        for prod_ in product(range(holdingPeriod + 1), range(-stopLoss, 1), [-1, 1]):
            count += 1
            trades_ = getTrades(series, dates, refDates_, prod_[0], prod_[1] * sigma, \
                                prod_[2])
            dates_, pnl_ = [j[0] for j in trades_], [j[3] for j in trades_]
            # 5) Eval performance
            if len(pnl_) > 2:
                # 6) Reconcile PnL
                pnl = attachTimeSeries(pnl, pnl_, dates_, count)
                # 7) Evaluate
                sr_, psr_, freq_ = evalPerf(pnl_, dates[0], dates[-1])
                for j in range(1, len(pnl_)): pnl_[j] += pnl_[j - 1]
                if sr == None or sr_ > sr:
                    psr, sr, trades = psr_, sr_, trades_
                    freq, pDay, prod = freq_, pDay_, prod_
                    print
                    count, pDay, prod, round(sr, 2), \
                    round(freq, 2), round(psr, 2)
    print
    'Total # iterations=' + str(count)
    return pnl, psr, sr, trades, freq, pDay, prod, dates, series


# ---------------------------------------------------------------
# Boilerplate
if __name__ == '__main__': backTest()
