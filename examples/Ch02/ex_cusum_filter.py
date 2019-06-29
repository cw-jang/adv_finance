
from adv_finance import datasets, stats, labeling


if __name__ == "__main__":
    ret = datasets.get_sample_returns(sigma=0.15)

    daily_vol = stats.get_daily_vol(ret, 20)
    threshold = daily_vol.ewm(20).mean() * 0.5
    side_events = labeling.cusum_filter(ret, threshold)


    print('Finished')

