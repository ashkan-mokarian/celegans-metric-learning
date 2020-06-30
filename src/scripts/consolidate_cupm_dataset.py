"""Creates a processed dataset for consistent matches between two worms based on pairwise matching solution files

--input_path: path to .sol files for both directions for all pair of worms in kolmogorov format
--output_file: path to output .npy file

the resulted pickle file consists of:
    - {w1_uid: {w2_uid: {w1_ul: w2_ul}}}
"""
import os
import argparse
import logging
import pickle
import pprint

from lib import utils
import lib.data.worms
import lib.data.labels
import lib.data.comb_pm as comb_pm

# noinspection PyUnresolvedReferences
import _init_paths

logger = logging.getLogger(__name__)


def get_consistent_upm(w1_w2_upm, w2_w1_upm):
    cupm = {w1ul:w2ul for w1ul, w2ul in w1_w2_upm.items() if w2ul in w2_w1_upm and w2_w1_upm[w2ul] == w1ul}
    return cupm


def get_upm(w1_w2_pm_file, w2_w1_pm_file, w1_nuclei_names_file, w2_nuclei_names_file, ulabels):
    w1_w2_pm = comb_pm.read_pm_sol(w1_w2_pm_file)
    w2_w1_pm = comb_pm.read_pm_sol(w2_w1_pm_file)
    w1_nuclei_names = comb_pm.read_nuclei_names(w1_nuclei_names_file)
    w2_nuclei_names = comb_pm.read_nuclei_names(w2_nuclei_names_file)

    w1_w2_upm = comb_pm.get_pm_ulabels(w1_w2_pm, w1_nuclei_names, w2_nuclei_names, ulabels)
    w2_w1_upm = comb_pm.get_pm_ulabels(w2_w1_pm, w2_nuclei_names, w1_nuclei_names, ulabels)

    return w1_w2_upm, w2_w1_upm


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
    if args.input_nuclei_names_path is not None:
        config['path'].update({'input_pm_nuclei_names': args.input_nuclei_names_path})
    if args.output_file is not None:
        config['path'].update({'cupm_dataset': args.output_file})
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

    # For every pair of worms, read the corresponding two solution files, change
    # label uid according to the standard uid extracted from universe.txt, only
    # keep consistent matchings, and save results. only keep a mapping from the
    # worm with the lower index to the higher one.

    solfile_name_pattern = '{}-to-{}.surf-18-10-22-1446.sol'
    nucleinamefile_name_pattern = '{}-NucleiNames.txt'

    worms = lib.data.worms.Worms(config['path']['worm_names'])
    ulabels = lib.data.labels.Labels(config['path']['universe_labels'])

    num_worms = len(worms._worm_names)

    w2w_cupm = dict()

    # pm: pairwise matching, upm: unique labeled pm, cupm: consistent upm

    for i in range(num_worms-1):
        for j in range(i+1, num_worms):
            w1name = worms.uid_to_name(i)
            w2name = worms.uid_to_name(j)
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

            logging.debug('w2w_consistent_pm:{}-{}: Files:\n\t[{}]\n\t[{}]\n\t[{}]\n\t[{}]'.format(
                i, j, w1_w2_pm_file, w2_w1_pm_file, w1_nuclei_names_file, w2_nuclei_names_file))

            w1_w2_upm, w2_w1_upm = get_upm(w1_w2_pm_file,
                w2_w1_pm_file, w1_nuclei_names_file, w2_nuclei_names_file, ulabels)

            tmp_w2w_cupm = get_consistent_upm(w1_w2_upm, w2_w1_upm)

            assert w2w_cupm.get(i,{}).get(j) is None

            w2w_cupm.update({i:{j:tmp_w2w_cupm}})

            logging.info('w2w_cupm:{}-{}: (direction)#match:#correct    (->){}:{} , (<-){}:{} , (<->){}:{}'.
                         format(i, j, len(w1_w2_upm), get_correct_matches(w1_w2_upm), len(w2_w1_upm),
                                get_correct_matches(w2_w1_upm), len(tmp_w2w_cupm), get_correct_matches(tmp_w2w_cupm))
                         )

    if os.path.exists(config['path']['consistent_pairwise_matching_dataset']) and not config['general']['overwrite']:
        logger.info('output file already exists. Turn on overwrite to overwrite. File: {}'.
                    format(config['path']['consistent_pairwise_matching_dataset']))
    else:
        with open(config['path']['consistent_pairwise_matching_dataset'], 'wb+') as f:
            pickle.dump(w2w_cupm, f)


if __name__ == '__main__':
    main()