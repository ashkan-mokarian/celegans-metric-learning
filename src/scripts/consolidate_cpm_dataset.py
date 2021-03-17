"""Creates a processed dataset for consistent matches between two worms based on pairwise matching solution files
from Lisa Kruse's solution format.

--input_path: path to .sol files for both directions for all pair of worms in Lisa Kruse's format and not Kolmogorov
sol format. i.e. a list of assignments
--output_file: path to output .npy file

the resulted pickle file consists of:
    - {'w1_uid-w2_uid': {w1_sh: w2_sh}}  uid = unique id for worms based on worm_names.txt, sh=segmentation hypothesis
"""
import os
import argparse
import logging
import pickle
import pprint

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


def get_config_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True, action='append',
                        help='path to config file, containing default project worm_names.txt')
    parser.add_argument('-i', '--input_solutions_path', required=True, type=str,
                        help='path to the pairwise matching solutions')
    parser.add_argument('-o', '--output_file', type=str,
                        help='output file to store results. If not specified read cpm_dataset value from config')
    parser.add_argument('--no_evaluation', action='store_true',
                        help='turns off evaluation. by default, looks in default location for worms dataset to '
                             'extract nuclei names, and check with universe labels, to evaluate pm solutions.')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrites existing output file')

    args = parser.parse_args()

    # maintain only one config for the rest of the code, place the cli input
    # somewhere appropriate in config
    config = utils.get_config(args.config)
    config['path'].update({'input_pm_sols': args.input_solutions_path})
    if args.output_file is not None:
        config['path'].update({'cpm_dataset': args.output_file})
    config['general'].update({'overwrite': args.overwrite})
    config['general'].update({'evaluate': not args.no_evaluation})
    if config['general']['evaluate']:
        config['path'].update({'worms_raw_dataset': os.path.join(DEFAULT_PATH.DATA, 'raw',
                                                                 '30WormsImagesGroundTruthSeg')})

    return config


def main(config):
    output_root = os.path.dirname(os.path.abspath(config['path']['cpm_dataset']))
    os.makedirs(output_root, exist_ok=True)

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler(os.path.join(output_root, 'consolidate_cpm_dataset.log')),
            logging.StreamHandler()
            ]
        )

    logging.info('\n'+pprint.pformat(config))

    # For every pair of worms, read the corresponding two solution files. only keep consistent
    # matchings, and save results. only keep a mapping from the worm with the lower index to the higher one.

    solfile_name_pattern = 'worm{:02}-worm{:02}.txt'
    seghypnames_fn = os.path.join(config['path']['worms_raw_dataset'], 'groundTruthInstanceSeg',
                                   '{}.ano.curated.aligned.txt')

    worms = lib.data.worms.Worms(config['path']['worm_names'])
    ulabels = lib.data.labels.Labels(config['path']['universe_labels'])

    num_worms = len(worms._worm_names)

    w2w_cpm = dict()

    # pm: pairwise matching, cpm: consistent pm

    for i in range(num_worms-1):
        w1uid = i+1
        w1name = worms.uid_to_name(w1uid)
        for j in range(i+1, num_worms):
            w2uid = j+1
            w2name = worms.uid_to_name(w2uid)

            w1_w2_pm_file = os.path.join(
                config['path']['input_pm_sols'],
                solfile_name_pattern.format(w1uid, w2uid)
                )
            w2_w1_pm_file = os.path.join(
                config['path']['input_pm_sols'],
                solfile_name_pattern.format(w2uid, w1uid)
                )

            logging.debug('w2w_consistent_pm:{}-{}: Files:\n\t[{}]\n\t[{}]'.format(
                w1uid, w2uid, w1_w2_pm_file, w2_w1_pm_file))

            tmp_w1_w2_pm = read_pm_sol_listfrmt(w1_w2_pm_file)
            tmp_w2_w1_pm = read_pm_sol_listfrmt(w2_w1_pm_file)
            # Add +1 to all labels, since labels in seghyp (worms_dataset.hdf) starts from 1
            w1_w2_pm = {k+1: v+1 for k,v in tmp_w1_w2_pm.items()}
            w2_w1_pm = {k+1: v+1 for k,v in tmp_w2_w1_pm.items()}

            tmp_w2w_cpm = get_consistent_pm(w1_w2_pm, w2_w1_pm)

            # assert w2w_cpm.get(i, {}).get(j) is None
            assert w2w_cpm.get(f'{w1uid}-{w2uid}') is None
            w2w_cpm.update({f'{w1uid}-{w2uid}': tmp_w2w_cpm})

            # Evaluation or not
            if config['general']['evaluate']:
                w1_seghyplnames = read_seghyp_names(seghypnames_fn.format(w1name))
                w2_seghyplnames = read_seghyp_names(seghypnames_fn.format(w2name))
                logging.info(f'w2w_cpm:{w1uid}-{w2uid}: (direction)n_matching:n_correct_matchings '
                             f'(->){len(w1_w2_pm)}:'
                             f'{get_correct_matches(w1_w2_pm, w1_seghyplnames, w2_seghyplnames, ulabels)} , '
                             f'(<-){len(w2_w1_pm)}:'
                             f'{get_correct_matches(w2_w1_pm, w2_seghyplnames, w1_seghyplnames, ulabels)} , '
                             f'final consistent pm (<->){len(tmp_w2w_cpm)}:'
                             f'{get_correct_matches(tmp_w2w_cpm, w1_seghyplnames, w2_seghyplnames, ulabels)}'
                             )

            else:
                logging.info(f'w2w_cpm:{w1uid}-{w2uid}: (direction)n_matching '
                         f'(->){len(w1_w2_pm)} , '
                         f'(<-){len(w2_w1_pm)} , '
                         f'final consistent pm (<->){len(tmp_w2w_cpm)}'
                         )

    if os.path.exists(config['path']['cpm_dataset']) and not config['general']['overwrite']:
        logger.info('output file already exists. Turn on overwrite to overwrite. File: {}'.
                    format(config['path']['cpm_dataset']))
    else:
        with open(config['path']['cpm_dataset'], 'wb+') as f:
            pickle.dump(w2w_cpm, f)


def helper_get_pm_and_acc(i, j):
    solfile_root = os.path.abspath(
        os.path.join(DEFAULT_PATH.DATA,
                     'raw',
                     '2020-07-22_w2w-solutions_3xLAP_1xTKR1000_Fusion')
        )
    solfile_name_pattern = 'worm{:02}-worm{:02}.txt'.format(i,j)
    solfile = os.path.join(solfile_root, solfile_name_pattern)
    worms = lib.data.worms.Worms(DEFAULT_PATH.WORM_NAMES)
    ulabels_fn = os.path.abspath(
        os.path.join(DEFAULT_PATH.DATA, 'raw', '30WormsImagesGroundTruthSeg', 'universe.txt')
        )
    ulabels = lib.data.labels.Labels(ulabels_fn)
    raw_worms_dataset_root = os.path.abspath(
        os.path.join(DEFAULT_PATH.DATA, 'raw', '30WormsImagesGroundTruthSeg', )
        )
    seghypnames_fn = os.path.join(raw_worms_dataset_root, 'groundTruthInstanceSeg',
                                  '{}.ano.curated.aligned.txt')
    w1name = worms.uid_to_name(i)
    w2name = worms.uid_to_name(j)

    tmp_w1_w2_pm = read_pm_sol_listfrmt(solfile)
    w1_w2_pm = {k + 1: v + 1 for k, v in tmp_w1_w2_pm.items()}

    w1_seghyplnames = read_seghyp_names(seghypnames_fn.format(w1name))
    w2_seghyplnames = read_seghyp_names(seghypnames_fn.format(w2name))

    acc = get_correct_matches(w1_w2_pm, w1_seghyplnames, w2_seghyplnames, ulabels)

    return w1_w2_pm, acc


if __name__ == '__main__':
    config = get_config_arguments()
    main(config)