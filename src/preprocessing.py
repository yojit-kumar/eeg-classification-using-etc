import os
import mne

def change_labels(raw1, raw2):
    """Standardize channel names and set montage."""
    labels = raw1.ch_names
    new_names = {}
    
    for ch in labels:
        clean_ch = ch.strip('.')
        if 'Fp' not in clean_ch:
            clean_ch = clean_ch.upper()
        if clean_ch.endswith('Z'):
            clean_ch = clean_ch[:-1] + 'z'
        new_names[ch] = clean_ch

    for raw in [raw1, raw2]:
        raw.rename_channels(new_names)
        raw.set_montage("standard_1020", verbose=False)

def extract_data(volunteer_id, root_dir):
    v_path = os.path.join(root_dir, volunteer_id)
    task1 = os.path.join(v_path, f'{volunteer_id}R01.edf')
    task2 = os.path.join(v_path, f'{volunteer_id}R02.edf')

    r1 = mne.io.read_raw_edf(task1, preload=True, verbose=False)
    r2 = mne.io.read_raw_edf(task2, preload=True, verbose=False)

    #Changing channel labels to standard names
    change_labels(r1, r2)
    ###########################################

    #Crop to get 60 seconds
    if r1.times[-1] > 60:
        r1.crop(tmin=0,tmax=60)
    if r2.times[-1] > 60:
        r2.crop(tmin=0,tmax=60)
    #######################

    return r1, r2



def filter_data(r1, r2):
    r1.notch_filter(freqs=60, verbose=False)
    r1_alpha = r1.copy().filter(8., 12., verbose=False)
    r2.notch_filter(freqs=60, verbose=False)
    r2_alpha = r2.copy().filter(8., 12., verbose=False)

    return r1_alpha, r2_alpha


if __name__ == '__main__':
    volunteer_ids = [f"S{n:03d}" for n in range(1,110)]
    root_dir = '../data/files/'
    result_dir = '../results/tmp/preprocessing/'

    for v in volunteer_ids:
        raw1, raw2 = extract_data(v, root_dir)
        raw1, raw2 = filter_data(raw1, raw2)

        result_path1 = os.path.join(result_dir, f'{v}R01.edf')
        result_path2 = os.path.join(result_dir, f'{v}R02.edf')
        mne.export.export_raw(result_path1, raw1, overwrite=True)
        mne.export.export_raw(result_path2, raw2, overwrite=True)





