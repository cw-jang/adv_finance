import sys
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import datetime as dt

import adv_finance.sampling as sampling
from adv_finance.multiprocess import process_jobs_, process_jobs
# from adv_finance.sampling import get_ind_matrix, get_avg_uniqueness


def get_rnd_t1(num_obs, num_bars, max_h):
    t1 = pd.Series()
    for i in np.arange(num_obs):
        ix = np.random.randint(0, num_bars)
        val = ix + np.random.randint(1, max_h)
        t1.loc[ix] = val
    return t1.sort_index()


def auxMC(num_obs, num_bars, max_h):
    t1 = get_rnd_t1(num_obs, num_bars, max_h)
    bar_idx = range(t1.max() + 1)

    ind_m = sampling.get_ind_matrix(bar_idx, t1)
    phi = sampling.seq_bootstrap(ind_m)
    seq_u = sampling.get_avg_uniqueness(ind_m[:, phi], None).mean()
    phi = np.random.choice(np.arange(ind_m.shape[1]), size=ind_m.shape[1])
    std_u = sampling.get_avg_uniqueness(ind_m[:, phi], None).mean()

    return {'std_u': std_u, 'seq_u': seq_u}


def mainMC(num_obs=10, num_bars=100, max_h=5, num_iters=5, num_threads=1):
    jobs = []
    for i in np.arange(num_iters):
        job = {'func': auxMC, 'num_obs': num_obs, 'num_bars': num_bars, 'max_h': max_h}
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads)

    return pd.DataFrame(out)


if __name__ == "__main__":
    # df = mainMC(numObs=10, numBars=10, numIters=20, numThreads=1)
    df = mainMC(num_obs=10, num_bars=10, num_iters=5, num_threads=2)

    print('finished')


### APPENDIX ###

# 원본 소스
# def seq_bootstrap_(ind_m, s_length=None):
#     if s_length is None:
#         s_length = ind_m.shape[1]
#
#     phi = []
#     while len(phi) < s_length:
#         c = ind_m[phi].sum(axis=1) + 1
#         avg_u = get_avg_uniqueness(ind_m, c)
#         prob = (avg_u / avg_u.sum()).values
#         phi += [np.random.choice(ind_m.columns, p=prob)]
#     return phi
#
#
# # Sparse Matrix 버전
# def seq_bootstrap(ind_m, s_length=None):
#     if s_length is None:
#         s_length = ind_m.shape[1]
#
#     phi = []
#     m = ind_m.todense()
#     while len(phi) < s_length:
#         m_ = m[:, phi]
#         c = m_.sum(axis=1) + 1
#         avg_u = sampling.get_avg_uniqueness(m, c)
#         prob = (avg_u / avg_u.sum())
#         prob = np.asarray(prob).reshape(-1)
#         phi += [np.random.choice(np.arange(ind_m.shape[1]), p=prob)]
#     return phi


# def expandCall(kargs):
#     # Expand the arguments of a callback function, kargs['func'] func= kargs['func']
#     func = kargs['func']
#     del kargs['func']
#     out= func (**kargs)
#     return out
#
#
# # single-thread execution for debugging [20.8]
# def processJobs_(jobs):
#     # Run jobs sequentially, for debugging
#     out=[]
#     for job in jobs:
#         out_= expandCall(job)
#         out.append(out_)
#     return out
#
# def report_progress(job_num, num_jobs, time0, task):
#     """
#     Snippet 20.9.1, pg 312, Example of Asynchrounous call to python multiprocessing library
#
#     :param job_num:
#     :param num_jobs:
#     :param time0:
#     :param task:
#     :return:
#     """
#
#     # Report progress as asynch jobs are completed
#     msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
#     msg.append(msg[1] * (1 / msg[0] - 1))
#     time_stamp = str(dt.datetime.fromtimestamp(time.time()))
#
#     msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '%' + task + ' done after ' + \
#         str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
#
#     if job_num < num_jobs:
#         sys.stderr.write(msg + '\r')
#     else:
#         sys.stderr.write(msg + '\n')

# def processJobs(jobs,task =None, numThreads=24):
#     # Run in parallel.
#     # jobs must contain a 'func' callback, for expandCall
#     if task is None:task = jobs [0]['func'].__name__
#     pool=mp. Pool(processes=numThreads)
#     outputs,out, time0 =pool . imap_unordered(expandCall,jobs) ,[], time. time()
#     # Process asyn output, report progress
#     for i,out_ in enumerate(outputs,1):
#         out.append(out_)
#         report_progress(i,len( jobs),time0, task)
#     pool.close()
#     pool.join()  # this is needed to prevent memory leaks
#     return out

# def auxMC_(numObs, numBars, maxH):
#     num_obs = numObs
#     num_bars = numBars
#     max_h = maxH
#
#     t1 = get_rnd_t1(num_obs, num_bars, max_h)
#     bar_idx = range(t1.max() + 1)
#     ind_m = get_ind_matrix(bar_idx, t1)
#     phi = np.random.choice(ind_m.columns, size=ind_m.shape[1])
#     std_u = get_avg_uniqueness(ind_m[phi]).mean()
#     phi = seq_bootstrap_(ind_m)
#     seq_u = get_avg_uniqueness(ind_m[phi]).mean()
#     return {'std_u': std_u, 'seq_u': seq_u}
#

# def mainMC(numObs=10, numBars=1000, maxH=5, numIters=1E3, numThreads=8):
#     # Monte Carlo experiments
#     jobs = []
#     for i in np.arange(numIters):
#         job = {'func':auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
#         jobs.append(job)
#
#     if numThreads == 1:
#         out = processJobs_(jobs)
#     else:
#         out = processJobs(jobs, numThreads=numThreads)
#
#     return pd.DataFrame(out)

# def get_ind_matrix(bar_idx, t1):
#     ind_m = pd.DataFrame(0, index=bar_idx,
#                          columns=range(t1.shape[0]))
#     for  i, (t0_, t1_) in enumerate(t1.iteritems()):
#         ind_m.loc[t0_:t1_, i] = 1
#     return ind_m
#
#
# def get_avg_uniqueness(ind_m, c=None):
#     if c is None:
#         c = ind_m.sum(axis=1)
#
#     ind_m = ind_m.loc[c > 0]
#     c = c.loc[c > 0]
#     u = ind_m.div(c, axis=0)
#     avg_u = u[u>0].mean()
#     avg_u = avg_u.fillna(0)
#     return avg_u
