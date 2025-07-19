import ETC
import pandas as pd


def etc_func(signal):
    seq = ETC.partition(signal, n_bins=2)
    res = ETC.compute_1D(seq, verbose=False).get('NETC1D')
    return res

    
