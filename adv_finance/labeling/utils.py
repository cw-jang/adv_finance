import numbers
from scipy.stats import norm


def get_gaussian_betsize(prob, num_classes=2):
    """Translate probability to bettingsize

    Params
    ------
    prob: array-like
    num_classes: int, default 2

    Returns
    -------
    array-like
    """
    if isinstance(prob, numbers.Number):
        if prob != 0 and prob != 1:
            signal = (prob - 1. / num_classes) / (prob * (1 - prob))
        else:
            signal = 2 * prob - 1
    else:
        signal = prob.copy()
        signal[prob == 1] = 1
        signal[prob == 0] = -1
        cond = (prob < 1) & (prob > 0)
        signal[cond] = (prob[cond] - 1. / num_classes) / (prob[cond] *
                                                          (1 - prob[cond]))
    return 2 * norm.cdf(signal) - 1