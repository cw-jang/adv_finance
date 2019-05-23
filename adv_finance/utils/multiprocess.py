"""
Contains the logic from chapter 20 on multiprocessing and vectorization
"""

import numpy as np
import pandas as pd


# Snippet 20.5 (page 306), the lin_parts function
def lin_parts(num_atoms, num_threads):
    """
    Snippet 20.5 (page 306), the lin_parts function

    The simplest way to form molecules is to partition a list of atoms in subsets of equal size,
    where the number of subsets is the minimum between the number of processors and the number
    of atoms. For N subsets we need to find the N+1 indeices that enclose the partitions.
    This logic is demonstrated in Snippet 20.5.

    This function partitions a list of atoms in subsets (molecules) of equal size.
    An atom is a set of indivisible set of tasks.

    :param num_atoms:
    :param num_threads:
    :return:
    """

    # Partition of atoms with a single loop
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# Snippet 20.8, pg 311, Single thread execution, for debugging
def process_jobs_(jobs):
    """
    # Snippet 20.8, pg 311, Single thread execution, for debugging

    Run jobs sequentially, for debugging

    :param jobs:
    :return:
    """
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)

    return out


# Snippet 20.10 Passing the job (molecule) to the callback function
def expand_call(kargs):
    """
    Snippet 20.10 Passing the job (molecule) to the callback function

    Expand the arguments of a callback function, kargs['func']

    :param kargs:
    :return:
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


# Snippet 20.8 (page 310), The mpPandasObj, used at various points in the book
def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, **kargs):
    """
    Snippet 20.7 (page 310), The mpPandasObj, used at various points in the book

    Parallelize jobs, return a dataframe or series.
    Example: df1=mp_pandas_obj(func, ('molecule', df0.index), 24, **kwds)

    First, atoms are grouped into molecules, using linParts (equal number of atoms per molecule)
    or nestedParts (atoms distributed in a lower-triangular structure). When mpBatches is greater
    than 1, there will be more molecues than cores. Suppose that we divide a task into 10 molecules,
    where molecule 1 takes twice as long as the rest. If we run this process in 10 cores, 9 of the
    cores will be idle half of the runtime, wating for the first core to process molecule 1.
    Alternatively, we could set mpBatches=10 so as to divide that task in 100 molecules. In doing so,
    every core will receive equal workload, even though the first 10 molecules take as much time as the
    next 20 molecules. In this example, the run with mpBatches=10 will take half of the time consumed by
    mpBatches=1.

    Second, we form a list of jobs. A job is a dictionary containing all the information needed to process
    a molecule, that is, the callback function, its keyword arguments, and the subset of atoms that form
    the molecule.

    Third, we will process the jobs sequentially if numThread==1 (see Snippet 20.8), and in parallel
    otherwise (see Section 20.5.2). The reason that we want the option to run jobs sequentially is for
    debugging purposes. It is not easy to catch a bug when programs are run in multiple processors.
    Once the code is debugged, we will want to use numThreads > 1.

    Fourch, we stitch together the output from every molecule into a single list, series, or dataframe.

    :param func: A callback function, which will be executed in parallel
    :param pd_obj: (tuple) Element 0: The name of the argument used to pass molecules to the callback function
        Element 1: A list of indivisible tasks(atoms), which will be grouped into molecules
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param mp_batches: (int) Number of parallel batches (jobs per core)
    :param lin_mols: (bool) Tells if the method should use linear or nested partitioning
    :param kargs: (var args) Keyword arguments needed by func
    :return: (data frame) of results
    """

    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    # else:
    #     parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_(jobs)
    # else:
    #     out = process_jobs(jobs, num_threads=num_threads)


    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out

    for i in out:
        df0 = df0.append(i)

    df0 = df0.sort_index()
    return df0