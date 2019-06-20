import numpy as np, scipy.stats as ss, pandas as pd, datetime as dt

from random import gauss
from sklearn.ensemble import RandomForestClassifier
from adv_finance.model_selection import kfold


def numBusinessDays(date0, date1):
    m, date0_ = 1, date0
    while True:
        date0_ += dt.timedelta(days=1)
        if date0_ >= date1: break
        if date0_.isoweekday() < 6: m += 1
    return m


if __name__ == "__main__":
    print("Started")

    # X, label = get_cls_data(n_features=10, n_informative=5, n_redundant=0, n_samples=1000)

    length = 500
    sigma = 1

    # Prepare series
    date_ = dt.date(year=2000, month=1, day=1)
    dates = []


    while len(dates) < length:
        # Monday == 1 ... Sunday == 7
        if date_.isoweekday() < 5:
            dates.append(date_)
        date_ += dt.timedelta(days=1)

    # 특정 투자 전략에 대한 임의의 daily return 값 데이터인 series를 생성
    series = np.empty((length))
    for i in range(series.shape[0]):
        series[i] = gauss(0, sigma)
        pDay_ = dt.date(year=dates[i].year, month=dates[i].month, day=1)

    ret = pd.DataFrame(series, index=dates)
    clf = RandomForestClassifier(oob_score=True, n_estimators=20)

    res, test_times = kfold.generate_signals(clf, ret)

    print("Finished")