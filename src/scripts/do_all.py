"""Script to do train-cluster-evaluate all together"""

import argparse
import logging
import os

import _init_paths

from train import main as train_main
from cluster import main as cluster_main
from evaluate import main as evaluate_main

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True, action='append',
                        help='path to config file, or name of configs to load sequentially, if not path and only ' \
                             'config name, assumes default dir structure')
    parser.add_argument('-n', '--name', type=str,
                        help='name of training session, used for some namings')
    parser.add_argument('--debug', action='store_true',
                        help='loads train_debug.toml for debug train mode')
    parser.add_argument('--load_last', action='store_true',
                        help='loads last saved model by searching in the experiments saved models')
    parser.add_argument('--load_best', action='store_true',
                        help='loads best saved model by searching in the experiments saved models')
    parser.add_argument('--append', action='store_true',
                        help='forces the exact name in the config file to be used without creating on-the-fly run-id '
                             'name if the experiment already exist. useful for when toml config file is created first in the experiment directory and not in the experiments_cfg')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    experiment_root = train_main(args)
    # after training, use the created toml config file for the rest
    args.config = os.path.join(experiment_root, 'train.toml')
    args.model = None
    args.only_best = False
    cluster_main(args)
    evaluate_main(args)