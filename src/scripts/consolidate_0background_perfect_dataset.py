"""Similar to consolidate_worms_dataset.py, creates the hdf dataset. This one creates so called perfect datasets.

I have two ideas for this. One is to keep the original size, collect intersection of all labeled data in the dataset,
and keep only the labels that are shared on all 30 worms. only keep raw values of pixels belonging to this
perfect_valid_list, 0 out everywhere.
"""
import h5py
import os
import argparse
import logging
import pprint
import skimage.io as io
import numpy as np
from scipy import ndimage

# noinspection PyUnresolvedReferences
import _init_paths

from lib import utils
import lib.data.worms
import lib.data.labels

logger = logging.getLogger(__name__)


def to_array(fn):
    image = io.imread(fn, plugin='simpleitk')
    return image


def get_config_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True, action='append',
                        help='path to config file, see example at experiments_cfg/consolidate_worms_dataset.toml')

    args = parser.parse_args()

    # maintain only one config for the rest of the code, place the cli input
    # somewhere appropriate in config
    config = utils.get_config(args.config)
    return config


def main(config):
    """generates .hdf datasets per worm (e.g. 01.hdf, 02.hdf, ...). numbers match with worm_names.txt

    Params:
    -------
    config: (dict) has following keys:
        ... keys mentioned in experiments_cfg/consolidate_worms_dataset.toml
    """
    # set logger
    os.makedirs(os.path.abspath(config['path']['worms_dataset']), exist_ok=True)
    # noinspection PyArgumentList
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=10,
        handlers=[
            logging.FileHandler(os.path.join(config['path']['worms_dataset'], 'consolidate_worms_dataset.log')),
            logging.StreamHandler()
            ]
        )

    logger.info('Starting consolidate_smaller_perfect_dataset with following config:')
    logger.info('\n' + pprint.pformat(config))

    # For every worm, read and compute only necessary data and store in one .hdf file, replace all worm with their
    # unique id from worm_names.txt. Also replace unique labeling of nuclei from universe.txt only for gt data (to
    # get consistent label numbers across all worms, this way just checking if label_i==label_j gives tp).

    if config['data']['aligned_worms']:
        raw_fn = 'imagesAsMhdRawAligned/{}.mhd'
        label_fn = 'groundTruthInstanceSeg/{}.ano.curated.aligned.tiff'
        label_metadata_fn = 'groundTruthInstanceSeg/{}.ano.curated.aligned.txt'
    else:
        raw_fn = 'imagesAsMhdRaw/{}.mhd'
        label_fn = 'groundTruthInstanceSeg/{}.ano.curated.tiff'
        label_metadata_fn = 'groundTruthInstanceSeg/{}.ano.curated.txt'
    output_fn = 'worm{:02}.hdf'

    worms = lib.data.worms.Worms(config['path']['worm_names'])
    ulabels = lib.data.labels.Labels(config['path']['universe_labels'])

    # for perfect_dataset we need the intersection of all valid label names on all the dataset
    perfect_dataset_valid_labelnames = ulabels._labels
    for n in range(len(worms._worm_names)):
        wid = n+1
        wname = worms.uid_to_name(wid)
        labelmetadata_f = os.path.join(
            config['path']['raw_worms_root'],
            label_metadata_fn.format(wname)
            )
        valid_label_list = []
        with open(labelmetadata_f) as f:
            for line in f:
                parts = line.split(' ')
                seghyp_label = int(parts[0])
                nuclei_name = parts[1].strip().upper()
                if ulabels.is_valid_label(nuclei_name):
                    valid_label_list.append(nuclei_name)
        perfect_dataset_valid_labelnames = perfect_dataset_valid_labelnames.intersection(set(valid_label_list))
    logger.info(f'perfect_dataset_valid_label:[len={len(perfect_dataset_valid_labelnames)}]')

    for n in range(len(worms._worm_names)):
        wid = n+1  # Start from 1, to be consistent with Lisa s code.
        wname = worms.uid_to_name(wid)
        raw_f = os.path.join(
            config['path']['raw_worms_root'],
            raw_fn.format(wname)
            )
        label_f = os.path.join(
            config['path']['raw_worms_root'],
            label_fn.format(wname)
            )
        labelmetadata_f = os.path.join(
            config['path']['raw_worms_root'],
            label_metadata_fn.format(wname)
            )
        output_f = os.path.abspath(os.path.join(
            config['path']['worms_dataset'],
            output_fn.format(worms.name_to_uid(wname))
            ))

        if os.path.exists(output_f) and not config['general']['overwrite']:
            logger.info('output file already exists. Turn on overwrite to overwrite. File: {}'.format(output_f))
            break

        logger.info(f'wormname:[{wname}], wid:[{worms.name_to_uid(wname):2}], raw_file:[{raw_f}], labeldata_file:['
                    f'{label_f}], labelmetadata_file:[{labelmetadata_f}], output_file:[{output_f}]')

        raw = to_array(raw_f).astype(np.uint8)
        # get nuclei segmentation hypothesis mask, here labels don't have necessarily
        # any meaning but they correspond with label names from labelmetadata_f
        nuclei_seghyp = to_array(label_f).astype(np.uint16)
        logger.debug(f'max_label_seghyp:[{np.max(nuclei_seghyp)}] - len_unique_label_seghyp:[{len(np.unique(nuclei_seghyp))}]')

        # labeling the segmentation hypothesis with unique universe label ids, results in approx 330 labels. Read
        # corresponding .ano.curated.aligned.txt file, numbers correspond to labels, names must be checked for validity
        gt_nuclei_labels = np.zeros_like(nuclei_seghyp)
        zerobackground_raw = np.zeros_like(raw)
        zerobackground_nuclei_seghyp = np.zeros_like(nuclei_seghyp)

        valid_label_list = []
        with open(labelmetadata_f) as f:
            for line in f:
                parts = line.split(' ')
                raw_label = int(parts[0])
                nuclei_name = parts[1].strip().upper()
                if nuclei_name not in perfect_dataset_valid_labelnames:
                    continue
                nuclei_uid = ulabels.label_to_uid(nuclei_name)
                valid_label_list.append(nuclei_uid)
                zerobackground_raw[nuclei_seghyp==raw_label] = raw[nuclei_seghyp==raw_label]
                zerobackground_nuclei_seghyp[nuclei_seghyp==raw_label] = nuclei_seghyp[nuclei_seghyp==raw_label]
                gt_nuclei_labels[nuclei_seghyp==raw_label] = nuclei_uid

        logger.debug(f'n valid labels:[{len(valid_label_list)}], n gt_seghyp unique labels:['
                     f'{len(np.unique(zerobackground_nuclei_seghyp))}], n gt_label unique labels:[{len(np.unique(gt_nuclei_labels))}]')


        # TODO: find a workaround
        # try:
        #     original_umask = os.umask(0)
        #     os.makedirs(output_file, mode=0o777,exist_ok=True)
        # finally:
        #     os.umask(original_umask)
        # os.makedirs(output_file, exist_ok=True)  # have to create the directory manually, otherwise hdf5 error,
        # probably permission issues
        with h5py.File(output_f, 'w') as f:

            f.create_dataset(
                'raw',  # raw input, without any normalization etc. [140x140x1166] uint8 values
                data=zerobackground_raw,
                compression='gzip'
                )
            f.create_dataset(
                'gt_seghyp',  # [140x140x1166] uint16, instance segmentation of nuclei,
                # label numbers without any meaning, they just correspond with label names in
                # *.ano.curated.aligned.txt
                data=zerobackground_nuclei_seghyp,
                compression='gzip'
                )
            f.create_dataset(
                'gt_label',  # [140x140x1166] uint16, gt nuclei labels, here the labels
                # correspond to the universe.txt labels. in other words, every label instance has actually a
                # name and not only a segmentation. usually around 330 labels compared to 558 due to missing data
                data=gt_nuclei_labels,
                compression='gzip'
                )


if __name__ == '__main__':
    config = get_config_arguments()
    main(config)