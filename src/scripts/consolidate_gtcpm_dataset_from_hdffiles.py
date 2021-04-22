"""Creates a processed dataset for consistent matches between two worms based on ground truth
--output_file: path to output .npy file

the resulted pickle file consists of:
    - {'w1_uid-w2_uid': {w1_sh: w2_sh}}  uid = unique id for worms based on worm_names.txt, sh=segmentation hypothesis
"""
import os
import argparse
import logging
import pickle
import pprint
import h5py
import numpy as np
import random
# noinspection PyUnresolvedReferences
import _init_paths

from lib import utils
import lib.data.worms
import lib.data.labels
from lib.utils.textparser import read_pm_sol_listfrmt, read_seghyp_names
from settings import DEFAULT_PATH

logger = logging.getLogger(__name__)


def get_consistent_pm(w1_w2_pm, w2_w1_pm):
    cpm = {w1l:w2l for w1l, w2l in w1_w2_pm.items() if w2l in w2_w1_pm and w2_w1_pm[w2l] == w1l}
    return cpm


def get_correct_matches(w1_w2_pm, w1_seghyplnames, w2_seghyplnames, ulabels):
    tmp = dict()
    for k,v in w1_w2_pm.items():
        klabel = w1_seghyplnames[k-1]
        vlabel = w2_seghyplnames[v-1]
        if not ulabels.is_valid_label(klabel):
            continue
        if not ulabels.is_valid_label(vlabel):
            continue
        tmp.update({ulabels.label_to_uid(klabel): ulabels.label_to_uid(vlabel)})
    return len([1 for k, v in tmp.items() if k==v])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='path to config file, containing default project worm_names.txt')
    parser.add_argument('-o', '--output_file', type=str,
                        help='output file to store results. If not specified read cpm_dataset value from config')
    parser.add_argument('--accuracy', type=int, default=100,
                        help='manipulates gtcpm to get desired accuracy')
    parser.add_argument('--std_dev', type=int, default=10,
                        help='standard deviation for accuracy manipulation')

    args = parser.parse_args()

    return args


def main(args):
    output_root = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_root, exist_ok=True)

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=10,
        handlers=[
            logging.FileHandler(os.path.join(output_root, f'consolidate_gtcpm_dataset_from_hdffiles-acc='
                                                          f'{args.accuracy}-stddev={args.std_dev}.log')),
            logging.StreamHandler()
            ]
        )

    logging.info('\n'+pprint.pformat(args))

    # For every pair of worms, read the corresponding two hdf files. only keep consistent
    # matchings, and save results. only keep a mapping from the worm with the lower index to the higher one.
    w2w_cpm = dict()

    # pm: pairwise matching, cpm: consistent pm / here consistent no meaning

    for i in range(30-1):
        w1uid = i+1
        w1hdf_fn = os.path.join(args.input_path, f'worm{w1uid:02d}.hdf')

        for j in range(i+1, 30):
            w2uid = j+1
            w2hdf_fn = os.path.join(args.input_path, f'worm{w2uid:02d}.hdf')

            tmp_w2w_cpm = dict()

            logger.info(f'gtcpm for [{w1uid}-{w2uid}]')

            with h5py.File(w1hdf_fn, 'r') as f:
                seghyp1 = f['gt_seghyp'][()]  # dtype: uint16
                label1 = f['gt_label'][()]
            with h5py.File(w2hdf_fn, 'r') as f:
                seghyp2 = f['gt_seghyp'][()]  # dtype: uint16
                label2 = f['gt_label'][()]

            unique_labels1 = set(np.unique(label1))
            unique_labels2 = set(np.unique(label2))
            assert unique_labels1 == unique_labels2

            for gtl in unique_labels1:
                seg_label1 = np.unique(seghyp1[label1==gtl])
                assert len(seg_label1) == 1
                seg_label2 = np.unique(seghyp2[label2==gtl])
                assert len(seg_label2) == 1

                tmp_w2w_cpm[seg_label1[0]] = seg_label2[0]

            # decrease accuracy of tmp_w2w_cpm if needed
            if args.accuracy < 100:
                # dont wanna change background pm
                tmp_w2w_cpm.pop(0)
                # choose a random accuracy from normal distribution
                acc = np.random.normal(args.accuracy, args.std_dev)
                if acc > 100:
                    acc=100
                total_pm = len(tmp_w2w_cpm)
                to_change = int((100-acc)/100*total_pm)
                keys_tochange = list(tmp_w2w_cpm.keys())
                random.shuffle(keys_tochange)
                keys_tochange = keys_tochange[:to_change]
                while len(keys_tochange) > 0:
                    if len(keys_tochange) == 1:
                        key_to_pop = keys_tochange.pop()
                        tmp_w2w_cpm.pop(key_to_pop)
                    elif len(keys_tochange) > 1:
                        # choose if pop or swap values
                        p = random.uniform(0, 1)
                        if p<0.5:
                            key_to_pop = keys_tochange.pop()
                            tmp_w2w_cpm.pop(key_to_pop)
                        else:
                            key1 = keys_tochange.pop()
                            key2 = keys_tochange.pop()
                            val1 = tmp_w2w_cpm[key1]
                            val2 = tmp_w2w_cpm[key2]
                            update_dict = {key1:val2, key2:val1}
                            tmp_w2w_cpm.update(update_dict)
                # add background again
                tmp_w2w_cpm.update({0:0})

            assert w2w_cpm.get(f'{w1uid}-{w2uid}') is None
            w2w_cpm.update({f'{w1uid}-{w2uid}': tmp_w2w_cpm})


    with open(args.output_file, 'wb+') as f:
        pickle.dump(w2w_cpm, f)


if __name__ == '__main__':
    args = get_args()
    main(args)