"""Creates an easier way to work with the worms dataset"""
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
                        help='path to config file, containing default project'
                             'config to access universe.txt and worm_names.txt')
    parser.add_argument('-i', '--input_worms_path', required=True, type=str,
                        help='path to raw worms data (30WormsImagesGroundTruthSeg)')
    parser.add_argument('-o', '--output_path', type=str,
                        help='output path to create data')

    args = parser.parse_args()

    # maintain only one config for the rest of the code, place the cli input
    # somewhere appropriate in config
    config = utils.get_config(args.config)
    config['path'].update({'input_worms_path': args.input_worms_path})
    if args.output_path is not None:
        config['path'].update({'worms_dataset': args.output_path})
    return config


def main():
    config = get_config_arguments()

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=config['general']['logging']
        )

    logging.info('\n' + pprint.pformat(config))

    # For every worm, read and compute only necessary data and store in one .hdf file, replace all worm and nuclei
    # names with their unique ids.
    raw_fn = 'imagesAsMhdRawAligned/{}.mhd'
    label_fn = 'groundTruthInstanceSeg/{}.ano.curated.aligned.tiff'
    label_metadata_fn = 'groundTruthInstanceSeg/{}.ano.curated.aligned.txt'

    worms = lib.data.worms.Worms(config['path']['worm_names'])
    ulabels = lib.data.labels.Labels(config['path']['universe_labels'])

    for wid in range(len(worms._worm_names)):
        wname = worms.uid_to_name(wid)
        raw_f = os.path.join(
            config['path']['input_worms_path'],
            raw_fn.format(wname)
            )
        label_f = os.path.join(
            config['path']['input_worms_path'],
            label_fn.format(wname)
            )
        labelmetadata_f = os.path.join(
            config['path']['input_worms_path'],
            label_metadata_fn.format(wname)
            )

        raw = to_array(raw_f).astype(np.uint8)
        # get nuclei instance segmentation mask and their corresponding centers, here labels don't have any meaning
        nuclei_instance = to_array(label_f).astype(np.uint16)
        con_instance = ndimage.measurements.center_of_mass(
            1*(nuclei_instance>0), nuclei_instance, range(1, np.max(nuclei_instance)+1, 1))
        con_instance = np.vstack(con_instance)
        con_instance = np.vstack([np.zeros(3), con_instance])  # label ids always start from 1. 0 reserved background

        # labeling the instance segmentations with unique universe label ids, results in apprx 330 labels. Read
        # corresponding .ano.curated.aligned.txt file, numbers correspond to labels, names must be checked for validity
        gt_nuclei_labels = np.zeros_like(nuclei_instance)
        gt_con_labels = np.zeros((559, 3))

        valid_label_list = []
        with open(labelmetadata_f) as f:
            for line in f:
                parts = line.split(' ')
                raw_label = int(parts[0])
                nuclei_name = parts[1].strip().upper()
                if not ulabels.is_valid_label(nuclei_name):
                    continue
                nuclei_uid = ulabels.label_to_uid(nuclei_name)
                valid_label_list.append(nuclei_uid)
                gt_nuclei_labels[nuclei_instance==raw_label] = nuclei_uid
        con_valid = ndimage.measurements.center_of_mass(
            1*(gt_nuclei_labels>0), gt_nuclei_labels, valid_label_list)
        for no, valid_label in enumerate(valid_label_list):
            gt_con_labels[valid_label] = np.array(con_valid[no])

        output_file = os.path.join(config['path']['worms_dataset'], str(wid) + '.hdf')
        output_file = os.path.abspath(output_file)
        if os.path.exists(output_file) and not config['general']['overwrite']:
            logger.info('output file already exists. Turn on overwrite to overwrite. File: {}'.format(output_file))
            break
        else:
            # os.makedirs(output_file, exist_ok=True)  # have to create the directory manually, otherwise hdf5 error,
            # probably permission issues
            with h5py.File(output_file, 'w') as f:

                f.create_dataset(
                    'volumes/raw',  # raw input, without any normalization etc. [140x140x1166] uint8 values
                    data=raw,
                    compression='gzip'
                    )
                f.create_dataset(
                    'volumes/nuclei_instances',  # [140x140x1166] uint16, instance segmentation of nuclei,
                    # label numbers without any meaning
                    data=nuclei_instance,
                    compression='gzip'
                    )
                f.create_dataset(
                    'matrix/con_instances',  # [max(nuclei_instances)x3] float32, center of nuclei,
                    # each row corresponds to the label in nuclei_instances
                    data=con_instance,
                    compression='gzip'
                    )
                f.create_dataset(
                    'volumes/gt_nuclei_labels',  # [140x140x1166] uint16, gt nuclei labels, here the labels
                    # correspond to the universe.txt labels. in other words, every labeld instance has actually a
                    # name and not only a segmentation. usually around 330 labels compared to 558 due to missing data
                    data=gt_nuclei_labels,
                    compression='gzip'
                    )
                f.create_dataset(
                    'matrix/gt_con_labels',  # center of nuclei matrix corresponding to labels in gt_nuclei_laebls,
                    # [559x3] bcuz at most 558 nuclei, 0 left empty bcuz label uid starts from 1, most rows 0 bcuz of
                    # missing data
                    data=gt_con_labels,
                    compression='gzip'
                    )


if __name__ == '__main__':
    main()