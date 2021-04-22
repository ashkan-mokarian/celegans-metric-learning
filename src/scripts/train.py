import argparse
import logging
import os
import time
import sys
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# noinspection PyUnresolvedReferences
import _init_paths

from settings import Settings, DefaultPath
from lib.utils.general import generate_run_id
from lib.data.tio_worms_dataset import TrainTioWormsDataset
from lib.models.unet_validpushforcediscloss import UnetValidPushForceDiscLoss

from lib.utils.gpu_profile import trace_calls, set_gpu_profile_fn

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


def get_settings(args):
    # overwrite .toml settings with CLI settings
    sett = Settings(args.config)
    if args.name:
        sett.NAME = args.name
    if args.load_last:
        sett.MODEL.INIT_MODEL_LAST = True
    if args.load_best:
        sett.MODEL.INIT_MODEL_BEST = True

    assert sum(bool(x) for x in [sett.MODEL.INIT_MODEL_LAST, sett.MODEL.INIT_MODEL_BEST, sett.MODEL.INIT_MODEL_PATH]) <= 1, \
        'Cannot set path/best/last model saving together'

    DEFAULT_PATH = DefaultPath()
    if not sett.PATH.EXPERIMENT_ROOT:
        sett.PATH.EXPERIMENT_ROOT = DEFAULT_PATH.EXPERIMENTS
    if not sett.PATH.WORMS_DATASET:
        sett.PATH.WORMS_DATASET = DEFAULT_PATH.WORMS_DATASET

    return sett


def init_fn(worker_id):
    random.seed(worker_id+time.time())


def main(args):
    print(os.environ.keys())
    sett = get_settings(args)

    # experiment root
    experiment_root = os.path.join(sett.PATH.EXPERIMENT_ROOT, sett.NAME)
    load_model_path = None

    ckpts_root = os.path.join(experiment_root, 'ckpts')
    if sett.MODEL.INIT_MODEL_BEST:
        ckpts = [os.path.join(ckpts_root, f) for f in os.listdir(ckpts_root) if f.startswith('bestmodel') and
                 f.endswith('.pth')]
        assert len(ckpts)==1, f'Best model loading assigned, but no best models exist in [{ckpts_root}]'
        load_model_path = ckpts[0]
    elif sett.MODEL.INIT_MODEL_PATH:
        raise NotImplementedError
    elif sett.MODEL.INIT_MODEL_LAST:
        ckpts = [os.path.join(ckpts_root, f) for f in os.listdir(ckpts_root) if f.endswith('.pth')]
        load_model_path = max(ckpts, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        assert os.path.isfile(load_model_path)
    elif os.path.exists(experiment_root) and not args.append:
        run_id = generate_run_id()
        sett.NAME = sett.NAME + '-' + run_id
        experiment_root = os.path.join(sett.PATH.EXPERIMENT_ROOT, sett.NAME)
    os.makedirs(experiment_root, exist_ok=True)

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=10,
        handlers=[
            logging.FileHandler(os.path.join(experiment_root, 'train.log')),
            logging.StreamHandler()
            ]
        )

    # gpu mem debug
    if sett.GENERAL.GPU_DEBUG:
        set_gpu_profile_fn(experiment_root)
        os.environ["GPU_DEBUG"] = sett.GENERAL.GPU_DEBUG
        os.environ['TRACE_INTO'] = sett.GENERAL.GPU_DEBUG_TRACE_INTO
        sys.settrace(trace_calls)


    # Load seeds
    random.seed(sett.GENERAL.SEED)
    np.random.seed(sett.GENERAL.SEED)
    torch.manual_seed(sett.GENERAL.SEED)

    logger.info("Start Training.")
    logger.info("Setting:" + str(sett))
    sett.get_toml_dict(os.path.join(experiment_root, 'train.toml'))

    train_dataset = TrainTioWormsDataset(
        dataset_root=sett.PATH.WORMS_DATASET,
        sett=sett,
        transforms=None,
        debug=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=sett.DATA.N_WORKER,
        worker_init_fn=init_fn,
        batch_size=None,
        pin_memory=True,
        persistent_workers=False)  # TODO: for torch 1.7 throws error with pin memory and persistent worker. update  and set this to tru

    tb_logs_dir = os.path.join(experiment_root, 'tblogs')
    os.makedirs(tb_logs_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_logs_dir)

    model = UnetValidPushForceDiscLoss(sett=sett)
    mock_input = next(iter(train_loader))
    model.print_model_summary(mock_input, tb_writer)

    model_save_path = os.path.join(experiment_root, 'ckpts')
    os.makedirs(model_save_path, exist_ok=True)
    model.fit(train_loader, sett.TRAIN.N_STEP, sett.TRAIN.LEARNING_RATE, sett.TRAIN.WEIGHT_DECAY, sett.TRAIN.LR_DROP_FACTOR,
              sett.TRAIN.LR_DROP_PATIENCE, sett.TRAIN.MODEL_CKPT_EVERY_N_STEP, model_save_path,
              sett.TRAIN.RUNNING_LOSS_INTERVAL, sett.TRAIN.BURN_IN_STEP, tb_writer)

    tb_writer.close()
    logger.info('Finished Training!!!')

    return experiment_root


if __name__ == '__main__':
    args = get_args()
    main(args)
