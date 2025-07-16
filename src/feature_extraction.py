import ETC
import pandas as pd


def etc_func(signal):
    seq = ETC.partition(signal, n_bins=4)
    res = ETC.compute_1D(seq, verbose=False).get('NETC1D')
    return res

def etc_with_no_order(r1, r2, volunteer_id, channel_ids):
    results = []
    labels = r1.ch_names
    label_idx = {label: idx for idx, label in enumerate(labels)}

    r1_data = r1.get_data()
    r2_data = r2.get_data()

    for ch in channel_ids:
        idx = label_idx[ch]
        signal1 = r1_data[idx,:]
        signal2 = r2_data[idx,:]

        etc_1 = etc_func(signal1)
        etc_2 = etc_func(signal2)

        results.append({
            'volunteer': volunteer_id,
            'channel': ch,
            'ETC_EyesOpen': etc_1,
            'ETC_EyesClosed': etc_2,
            })
        
    df = pd.DataFrame(results)
    return df
    
