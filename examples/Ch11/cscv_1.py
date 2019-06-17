# http://quantresearch.org/CSCV_1.py.txt

# !/usr/bin/env python
# On 20130704 by lopezdeprado@lbl.gov
# ----------------------------------------------------------------------------------------
def testAccuracy_MC(sr_base, sr_case):
    # Test the accuracy of CSCV against hold-out
    # It generates numTrials random samples and directly computes the �
    # � proportion where OOS performance was below the median.
    length, numTrials, numMC = 1000, 100, 1000
    pathOutput = 'H:/Studies/Quant #23/paper/'
    # 1) Determine mu,sigma
    mu_base, sigma_base = sr_base / (365.25 * 5 / 7.), 1 / (365.25 * 5 / 7.) ** .5
    mu_case, sigma_case = sr_case / (365.25 * 5 / 7.), 1 / (365.25 * 5 / 7.) ** .5
    hist, probOverfit = [], 0
    # 2) Generate trials
    for m in range(numMC):
        for i in range(1, numTrials):
            j = np.array([gauss(0, 1) for j in range(length)])
            j *= sigma_base / np.std(j)  # re-scale
            j += mu_base - np.mean(j)  # re-center
            j = np.reshape(j, (j.shape[0], 1))
            if i == 1:
                pnl = np.copy(j)
            else:
                pnl = np.append(pnl, j, axis=1)
        # 3) Add test case
        j = np.array([gauss(0, 1) for j in range(length)])
        j *= sigma_case / np.std(j)  # re-scale
        j += mu_case - np.mean(j)  # re-center
        j = np.reshape(j, (j.shape[0], 1))
        pnl = np.append(pnl, j, axis=1)
        # 4) Run test
        # Reference distribution
        mu_is = [np.average(pnl[:length / 2, i]) for i in range(pnl.shape[1])]
        sigma_is = [np.std(pnl[:length / 2, i]) for i in range(pnl.shape[1])]
        mu_oos = [np.average(pnl[length / 2:, i]) for i in range(pnl.shape[1])]
        sigma_oos = [np.std(pnl[length / 2:, i]) for i in range(pnl.shape[1])]
        sr_is = [mu_is[i] / sigma_is[i] for i in range(len(mu_is))]
        sr_oos = [mu_oos[i] / sigma_oos[i] for i in range(len(mu_oos))]
        print
        m, sr_is.index(max(sr_is)), max(sr_is), \
        sr_oos[sr_is.index(max(sr_is))]
        sr_oos_ = sr_oos[sr_is.index(max(sr_is))]
        hist.append(sr_oos_)
        if sr_oos_ < np.median(sr_oos): probOverfit += 1
    probOverfit /= float(numMC)
    print
    probOverfit
    return
