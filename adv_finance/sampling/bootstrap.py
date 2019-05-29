import numpy as np

from scipy.sparse import csr_matrix
from tqdm import tqdm

def get_ind_matrix(bar_ix, t1, verbose=True):
    """
    Snippet 4.3 (page 64) Build an Indicator Matrix

    :param bar_ix:
    :param t1:
    :return:
    """
    try:
        n_row = len(bar_ix)
        n_col = len(t1)
        mat = csr_matrix((n_row, n_col), dtype='b')

        for i, (t0, t1) in tqdm(enumerate(t1.iteritems()), position=True, disable=not verbose):
            mat[t0:t1 + 1, i] = 1

    except Exception as e:
        print(e)

    return mat


def get_avg_uniqueness(ind_m, c):
    """
    Snippet 4.4 (page 65) Compute Average Uniqueness

    :param ind_mat:
    :return:
    """

    try:
        if c is None:
            c = ind_m.sum(axis=1)

        u = ind_m / c
        u[np.isnan(u)] = 0
        avg_u = u.sum(axis=0) / ind_m.sum(axis=0)
        return avg_u

    except Exception as e:
        print(e)

    return None


def seq_bootstrap(ind_m, s_length=None, verbose=True):
    """
    Snippet 4.5 (page 65) Return Sample From Sequential Bootstrap

    :param ind_m:
    :param s_length:
    :return:
    """
    if s_length is None:
        s_length = ind_m.shape[1]

    phi = []
    m = ind_m.todense()

    for i in tqdm(np.arange(s_length), position=0, disable=not verbose):
    # while len(phi) < s_length:
        m_ = m[:, phi]
        c = m_.sum(axis=1) + 1
        avg_u = get_avg_uniqueness(m, c)
        prob = (avg_u / avg_u.sum())
        prob = np.asarray(prob).reshape(-1)
        phi += [np.random.choice(np.arange(ind_m.shape[1]), p=prob)]
    return phi