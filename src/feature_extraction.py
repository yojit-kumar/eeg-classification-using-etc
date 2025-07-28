import ETC
import numpy as np
from itertools import permutations


def etc_func(signal, bins=2):
    seq = ETC.partition(signal, n_bins=bins)
    res = ETC.compute_1D(seq, verbose=False).get('NETC1D')
    return res

def permute_patterns(m):
    all_perms = list(permutations(range(m)))
    pattern_to_symbol = {perm: i for i, perm in enumerate(all_perms)}
    return pattern_to_symbol, len(all_perms)

def ordinal_patterns(pattern_to_symbol, ts, m, DELAY=1):
    """Convert time series to ordinal pattern symbols."""
    ts = np.array(ts)
    n = len(ts)
       
    symbols = []
    i = 0
    while i < n - m:
        window = ts[i:i+m]
        ordinal_pattern = tuple(np.argsort(window))
        symbol = pattern_to_symbol[ordinal_pattern]
        symbols.append(symbol)
        i += DELAY 
    
    return symbols
