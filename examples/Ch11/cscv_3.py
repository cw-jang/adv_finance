from __future__ import print_function

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

    # d값은 2000-01-01부터 dates[0]값 (2000-01-03)사이의 numBusinessDay 인 2부터 시작
    # d값을 하나씩 증가시키면서 pDay에 넣고 중간에 월이 바뀌면 d값은 1로 초기화
    # 결과적으로 pDay가 갖고 있는 값은 date[i]가 매월 N번째 BusinessDay에 해당하는지 그 오프셋 값을 담게 된다
    for i in dates:
        if m != i.month:
            # m, d = i.month, 1
            m = i.month
            d = 1

        pDay.append(d)
        d += 1

    for j in range(1, 30):
        # 매월의 각 날짜(1~30)들의 리스트를 만들고 각각 매월 N번째 BusinessDay 끼리 묶어준다
        # 즉, 매월 1번째 B-Day는 1번째 B-Day 끼리, 2번째는 2번째 B-Day끼리..  식으로
        # 예) dict[1] = [2000-02-01, 2000-03-01, ... ]
        # 예) dict[3] = [2000-01-04, 2000-02-03, 2000-03-06, ...]
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
    # trades, pnl, position_, j, num = [], 0, 0, 0, None
    trades = []
    pnl = 0
    position_ = 0
    j = 0
    num = None
    for i in range(1, len(dates)):
        # Find next roll and how many trading dates to it
        if dates[i] >= refDates[j]:
            # numl, pnl 세팅한지 하루는 지나야한다???, entry 날짜 하루를 확보하기 위함?
            #if dates[i] <= refDates[j]:, 즉 refDates[j]에 터치해서 들어왔을 경우(예: 2000-02-01) num, pnl reset
            if dates[i - 1] < refDates[j]:
                num, pnl = 0, 0

            if j < len(refDates) - 1:
                # dates[i]가 refDates[j]를 터치해서 entry가 들어왔을 경우 refDates[j]를 다시 advance 시켜준다
                while dates[i] > refDates[j]:
                    j += 1

        if num == None:
            continue

        # Trading rule
        position = 0

        # num 값이 exit(=holding_period)보다 작고 pnl이 stopLoss보다 크면 position = side 값
        # 즉, right & bottom barrier를 터치하지 않으면 position 은 side 값으로 할당
        if num < exit and pnl > stopLoss:
            position = side

        # position 값이 0이 아니거나(-1 or 1) postion_ 값이 0이 아닌 경우?
        # position_값은 이전의 position값을 담는 임시변수, 결국은 entry & exit 판단
        if position != 0 or position_ != 0:
            # date, num(=holding_period?), side, return 순으로 넣어준다
            trades.append( [dates[i], num, position, position_ * (series[i] - series[i - 1])] )

            # pnl 값에 return을 누적시켜준다
            pnl += trades[-1][3]

            # 현재 position값을 position_에 업데이트
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
    holdingPeriod = 20
    sigma = 1
    stopLoss = 10
    length = 1000

    # 2) Prepare series
    date_ = dt.date(year=2000, month=1, day=1)
    dates = []

    while len(dates) < length:
        # Monday == 1 ... Sunday == 7
        if date_.isoweekday() < 5: dates.append(date_)
        date_ += dt.timedelta(days=1)

    # 특정 투자 전략에 대한 임의의 daily return 값 데이터인 series를 생성
    series = np.empty((length))
    for i in range(series.shape[0]):
        series[i] = gauss(0, sigma)
        pDay_ = dt.date(year=dates[i].year, month=dates[i].month, day=1)
        # 거래소가 개장 안하는 공휴일에도 가격이 움직인다
        if numBusinessDays(pDay_, dates[i]) <= nDays:
            series[i] += sigma * factor
    series = np.cumsum(series)

    # 3) Optimize
    # 각각의 매월 N번째 B-Day끼리 따로 묶어서 refDates dict에 담는다
    refDates = getRefDates_MonthBusinessDate(dates)
    psr, sr, trades, sl, freq, pDay, pnl, count = None, None, None, None, None, None, None, 0
    for pDay_ in refDates.keys():
        refDates_ = refDates[pDay_]

        # B-Day가 없는 날은 Pass (예: 매월 29번째 B-Day)
        if len(refDates_) == 0:
            continue

        # 4) Get trades
        # holdingPeriod(0~20일) * -stopLoss(-10~0) * side(-1, 1) 조합의 경우의 수만큼 trial
        for prod_ in product(range(holdingPeriod + 1), range(-stopLoss, 1), [-1, 1]):

            # DEBUG: holdingPeriod값이 0인 경우는 trade가 발생하지 않으므로 디버깅에서 pass
            if prod_[0] <= 1:
                continue

            count += 1

            # 다음의 진입(Entry)/청산(Exit) 조건으로 Trade 레코드를 만든다
            #   - Entry:RefDates(매달 N번째 B-Day)
            #   - Exit:RefDates+HoldingPeriod or StopLoss
            trades_ = getTrades(series, dates, refDates_, prod_[0], prod_[1] * sigma, prod_[2])
            # dates_, pnl_ = [j[0] for j in trades_], [j[3] for j in trades_]
            dates_ = [j[0] for j in trades_]
            pnl_ = [j[3] for j in trades_]

            # 5) Eval performance
            if len(pnl_) > 2:
                # 6) Reconcile PnL
                # pnl DataFrame에 count를 label로 하는 pnl_ series를 추가한다
                pnl = attachTimeSeries(pnl, pnl_, dates_, count)
                # 7) Evaluate
                sr_, psr_, freq_ = evalPerf(pnl_, dates[0], dates[-1])
                for j in range(1, len(pnl_)):
                    pnl_[j] += pnl_[j - 1]
                if sr == None or sr_ > sr: # best sr을 찾는다
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
