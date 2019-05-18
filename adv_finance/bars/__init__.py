from numba import jit


@jit(nopython=True)
def mad_outlier(y, thresh=3.):
    '''
    compute outliers based on mad
    # args
        y: assumed to be array with shape (N,1)
        thresh: float()
    # returns
        array index of outliers
    '''
    median = np.median(y)
    diff = np.sum((y - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh



def select_sample_data(ref, sub, price_col, date): 
    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]
    return xdf, xtdf

def plot_sample_data(ref, sub, bar_type, *args, **kwds):
    f,axes=plt.subplots(3,sharex=True, sharey=True, figsize=(10,7))
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend();
    
    ref.plot(*args, **kwds, ax=axes[1], label='price', marker='o')
    sub.plot(*args, **kwds, ax=axes[2], ls='', marker='X',
             color='r', label=bar_type)

    for ax in axes[1:]: ax.legend()
    plt.tight_layout()
    
    return

def volume_bars(df, volume_col, n): 
    'compute volume bars'
    
    vols = df[volume_col]
    tick_n = 0
    idx = []
    for i, x in enumerate(tqdm(vols)): 
        tick_n += x 
        if tick_n >= n: 
            idx.append(i)
            tick_n = 0
            continue
            
    return df.iloc[idx]

def tick_bars(df, price_col, n): 
    '''
    compute tick bars 
    '''    
    prices = df[price_col]
    tick_n = 0
    idx = []
    for i, x in enumerate(tqdm(prices)): 
        tick_n += 1
        if tick_n >= n: 
            idx.append(i)
            tick_n = 0
            continue
    return df.iloc[idx]


def dollar_bars(df, dv_col, n): 
    '''
    compute dollar bars 
    '''
    t = df[dv_col]
    tick_n = 0
    idx = []
    for i, x in enumerate(tqdm(t)): 
        tick_n += x 
        if tick_n >= n: 
            idx.append(i)
            tick_n = 0
            continue
    
    return df.iloc[idx]
        
    
    
    