# Since using amira to get results is not easy, stick to Lisa's code as default, and keep this only if needed later

"""Creates a processed dataset for consistent matches between two worms based on pairwise matching solution files

consider the two worms w1='C18G1_2L1_1" and w2='cnd1threeL1_1213061'.
In WormEval-18-10-22/nuclei-names w1=566 w2=567 both starting from 1.
in 18-10-22-1446-sol exactly the same they just start from 0
In 30WormsImagesGroundTruthSeg/groundTruthInstanceSeg/*.ano.curated.aligned.txt w1=555 w2=554 the result of some curate
So, this scripts, removes assignments that are not present in *.ano.curated.aligned.txt since I don't have any
reference to the actual image. adds +1 to matching assignments since labels start from 1 and not 0. It also relabels
the assignments in 18-10-2201446-sol files such that they match the
label numbers in *.ano.curated.aligned.txt.
This way, a matching assignment w1:{w2:{10:15}} could be looked up by calling .hdf files for w1 and accessing
seghyp 10 in volumes/nuclei_seghyp (not volumes/gt_nuclei_labels)

--input_path: path to .sol files for both directions for all pair of worms in kolmogorov format
--output_file: path to output .npy file

the resulted pickle file consists of:
    - {'w1_uid-w2_uid': {w1_sh: w2_sh}}  uid = unique id for worms based on worm_names.txt, sh=segmentation hypothesis
"""
import os
import argparse
import logging
import pickle
import pprint

from lib import utils
import lib.data.worms
import lib.data.labels
from lib.utils.textparser import read_seghyp_names, read_pm_sol_kolmogorov

# noinspection PyUnresolvedReferences
import _init_paths

logger = logging.getLogger(__name__)


def get_consistent_pm(w1_w2_pm, w2_w1_pm):
    cpm = {w1l:w2l for w1l, w2l in w1_w2_pm.items() if w2l in w2_w1_pm and w2_w1_pm[w2l] == w1l}
    return cpm


def read_nuclei_names(file):
    nuclei_names = list()
    try:
        with open(file) as f:
            for line in f:
                nuclei_names.append(line.strip().upper())
    except FileNotFoundError:
        raise FileNotFoundError
    return nuclei_names


def relabel_pm(pm_sol, nuclei_names1, nuclei_names2, reference_names1, reference_names2):
    rpm = dict()
    for id1, id2 in pm_sol.items():
        name1 = nuclei_names1[id1]
        name2 = nuclei_names2[id2]
        if name1 in reference_names1 and name2 in reference_names2:
            rid1 = reference_names1.index(name1)
            rid2 = reference_names2.index(name2)
            rpm.update({rid1: rid2})
    return rpm


def get_rpm(w1_w2_pm_file, w2_w1_pm_file, w1_nuclei_names_file, w2_nuclei_names_file, w1_seghyp_names_file,
            w2_seghyp_names_file):
    w1_w2_pm = read_pm_sol_kolmogorov(w1_w2_pm_file)
    w2_w1_pm = read_pm_sol_kolmogorov(w2_w1_pm_file)
    w1_nuclei_names = read_nuclei_names(w1_nuclei_names_file)
    w2_nuclei_names = read_nuclei_names(w2_nuclei_names_file)
    w1_seghyp_names = read_seghyp_names(w1_seghyp_names_file)
    w2_seghyp_names = read_seghyp_names(w2_seghyp_names_file)

    w1_w2_rpm = relabel_pm(w1_w2_pm, w1_nuclei_names, w2_nuclei_names, w1_seghyp_names, w2_seghyp_names)
    w2_w1_rpm = relabel_pm(w2_w1_pm, w2_nuclei_names, w1_nuclei_names, w2_seghyp_names, w1_seghyp_names)

    return w1_w2_rpm, w2_w1_rpm


def get_correct_matches(upm):
    return len([1 for k, v in upm.items() if k==v])


def get_config_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True, action='append',
                        help='path to config file, containing default project'
                             'config to access universe.txt and worm_names.txt')
    parser.add_argument('-i', '--input_solutions_path', required=True, type=str,
                        help='path to the pairwise matching solutions')
    parser.add_argument('-i2', '--input_nuclei_names_path', type=str,
                        help='path to the nuclei names used in the solutions'
                             'path, defaults to the same -i path')
    parser.add_argument('-i3', '--input_nuclei_names_path_reference', required=True, type=str,
                        help='path to reference segmentation hypothesis names in '
                             '30WormsImagesGroundTruthSeg/groundTruthInstanceSeg. '
                             'looks for {wormname}.ano.curated.aligned.txt in the given path')
    parser.add_argument('-o', '--output_file', type=str,
                        help='output file to store results')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrites existing output file')

    args = parser.parse_args()

    # maintain only one config for the rest of the code, place the cli input
    # somewhere appropriate in config
    config = utils.get_config(args.config)
    config['path'].update({'input_pm_sols': args.input_solutions_path})
    config['path'].update({'input_pm_nuclei_names': args.input_solutions_path})
    config['path'].update({'input_reference_nuclei_names': args.input_nuclei_names_path_reference})
    if args.input_nuclei_names_path is not None:
        config['path'].update({'input_pm_nuclei_names': args.input_nuclei_names_path})
    if args.output_file is not None:
        config['path'].update({'cpm_dataset': args.output_file})
    config['general'].update({'overwrite': args.overwrite})

    return config


def main():
    # get arguments and overwrite configs if there is default config pattern
    config = get_config_arguments()

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=config['general']['logging']
            )

    logging.info('\n'+pprint.pformat(config))

    # For every pair of worms, read the corresponding two solution files, change label id according to the reference
    # segmentation hypothesis (seghyp) id extracted from {worm_name}.ano.curated.aligned.txt. only keep consistent
    # matchings, and save results. only keep a mapping from the worm with the lower index to the higher one.

    solfile_name_pattern = '{}-to-{}.surf-18-10-22-1446.sol'
    nucleinamefile_name_pattern = '{}-NucleiNames.txt'
    seghypfile_name_pattern = '{}.ano.curated.aligned.txt'

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
                solfile_name_pattern.format(w1name, w2name)
                )
            w2_w1_pm_file = os.path.join(
                config['path']['input_pm_sols'],
                solfile_name_pattern.format(w2name, w1name)
                )
            w1_nuclei_names_file = os.path.join(
                config['path']['input_pm_nuclei_names'],
                nucleinamefile_name_pattern.format(w1name)
                )
            w2_nuclei_names_file = os.path.join(
                config['path']['input_pm_nuclei_names'],
                nucleinamefile_name_pattern.format(w2name)
                )
            w1_seghyp_names_file = os.path.join(
                config['path']['input_reference_nuclei_names'],
                seghypfile_name_pattern.format(w1name)
                )
            w2_seghyp_names_file = os.path.join(
                config['path']['input_reference_nuclei_names'],
                seghypfile_name_pattern.format(w2name)
                )

            logging.debug('w2w_consistent_pm:{}-{}: Files:\n\t[{}]\n\t[{}]\n\t[{}]\n\t[{}]\n\t[{}]\n\t[{}]'.format(
                w1uid, w2uid, w1_w2_pm_file, w2_w1_pm_file, w1_nuclei_names_file, w2_nuclei_names_file,
                w1_seghyp_names_file,
                w2_seghyp_names_file))

            w1_w2_rpm, w2_w1_rpm = get_rpm(w1_w2_pm_file,
                w2_w1_pm_file, w1_nuclei_names_file, w2_nuclei_names_file, w1_seghyp_names_file, w2_seghyp_names_file)

            tmp_w2w_cpm = get_consistent_pm(w1_w2_rpm, w2_w1_rpm)

            # Add +1 to all labels, since labels in seghyp (worms_dataset.hdf) starts from 1
            tmptmp_w2w_cpm = {}
            for ii, jj in tmp_w2w_cpm.items():
                tmptmp_w2w_cpm.update({ii+1:jj+1})

            # assert w2w_cpm.get(i, {}).get(j) is None
            assert w2w_cpm.get(f'{w1uid}-{w2uid}') is None
            w2w_cpm.update({f'{w1uid}-{w2uid}': tmptmp_w2w_cpm})

            # Just for logging purpose
            w1_w2_pm = read_pm_sol_kolmogorov(w1_w2_pm_file)
            w2_w1_pm = read_pm_sol_kolmogorov(w2_w1_pm_file)

            logging.info(f'w2w_cpm:{i}-{j}: (direction)#match before:after relabeling   '
                         f'(->){len(w1_w2_pm)}:{len(w1_w2_rpm)} , '
                         f'(<-){len(w2_w1_pm)}:{len(w2_w1_rpm)} , '
                         f'final consistent pm (<->){len(tmptmp_w2w_cpm)}'
                         )

    if os.path.exists(config['path']['cpm_dataset']) and not config['general']['overwrite']:
        logger.info('output file already exists. Turn on overwrite to overwrite. File: {}'.
                    format(config['path']['cpm_dataset']))
    else:
        with open(config['path']['cpm_dataset'], 'wb+') as f:
            pickle.dump(w2w_cpm, f)


if __name__ == '__main__':
    main()