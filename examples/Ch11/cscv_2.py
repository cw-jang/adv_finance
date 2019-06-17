# http://quantresearch.org/CSCV_2.py.txt

#!/usr/bin/env python
# On 20130704 by lopezdeprado@lbl.gov
#----------------------------------------------------------------------------------------
def testAccuracy_EVT(sr_base, sr_case):
    # Test accuracy by numerical integration
    # It does the same as testAccuracy_MC, but through numerical integration ...
    # ... of the base and case distributions.
    # 1) Parameters
    parts, length, freq, minX, trials = 1e4, 1000, 365.25 * 5 / 7., -10, 100
    emc = 0.57721566490153286  # Euler-Mascheroni constant
    # 2) SR distributions
    dist_base = [sr_base, ((freq + .5 * sr_base ** 2) / length) ** .5]
    dist_case = (sr_case, ((freq + .5 * sr_case ** 2) / length) ** .5)
    # 3) Fit Gumbel (method of moments)
    maxList = []
    for x in range(int(parts)):
        max_ = max([gauss(dist_base[0], dist_base[1]) for i in range(trials)])
        maxList.append(max_)
    dist_base[1] = np.std(maxList) * 6 ** .5 / math.pi
    dist_base[0] = np.mean(maxList) - emc * dist_base[1]
    # 4) Integration
    prob1 = 0
    for x in np.linspace(minX * dist_case[1], 2 * dist_case[0] - sr_base, parts):
        f_x = ss.norm.pdf(x, dist_case[0], dist_case[1])
        F_y = 1 - ss.gumbel_r.cdf(x, dist_base[0], dist_base[1])
        prob1 += f_x * F_y
    prob1 *= (2 * dist_case[0] - sr_base - minX * dist_case[1]) / parts
    prob2 = 1 - ss.norm.cdf(2 * dist_case[0] - sr_base, dist_case[0], dist_case[1])
    print
    dist_base, dist_case
    print
    prob1, prob2, prob1 + prob2
    return
