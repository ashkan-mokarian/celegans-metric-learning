import h5py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths

from settings import DEFAULT_PATH

EXP_ROOT = '/home/ashkan/workspace/deployed/worms_nuclei_metric_learning-deployed/experiments/'
# EXP_ROOT += 'one_consistent-patch32-skip_large_lass'
EXP_ROOT += 'default'

STEP = 100000

# Z_SLICES = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
Z_SLICES = [50]

WORMS = [1, 2]


def main():
    worm1 = WORMS[0]
    worm2 = WORMS[1]
    if worm2 < worm1:
        worm1, worm2 = worm2, worm1

    worm_hdf_fn = os.path.join(DEFAULT_PATH.WORMS_DATASET, 'worm{:02}.hdf')
    worm1_f = worm_hdf_fn.format(worm1)
    worm2_f = worm_hdf_fn.format(worm2)

    with h5py.File(worm1_f, 'r') as f:
        raw1 = f['volumes/raw'][()].astype('float')
        seghyp1 = f['volumes/nuclei_seghyp'][()].astype('int')
        gt_label1 = f['volumes/gt_nuclei_labels'][()].astype('int')

    with h5py.File(worm2_f, 'r') as f:
        raw2 = f['volumes/raw'][()].astype('float')
        seghyp2 = f['volumes/nuclei_seghyp'][()].astype('int')
        gt_label2 = f['volumes/gt_nuclei_labels'][()].astype('int')

    plot_save_fn = os.path.join(EXP_ROOT, 'output', 'plots', 'pm', f'{worm1}-{worm2}'
                                    , f'step={STEP}-slice='+'{}.html')

    for slice in Z_SLICES:
        plot_save_f = plot_save_fn.format(slice)
        os.makedirs(os.path.dirname(plot_save_f), exist_ok=True)




if __name__ == '__main__':
    main()