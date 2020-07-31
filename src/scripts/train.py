import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# noinspection PyUnresolvedReferences
import _init_paths

from settings import Settings, DefaultPath
from lib.utils.general import generate_run_id
from lib.data.worms_dataset import WormsDataset
from lib.models.pixelwise_model import PixelwiseModel

logger = logging.getLogger(__name__)


def get_settings():
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
    args = parser.parse_args()

    # overwrite .toml settings with CLI settings
    sett = Settings(args.config)
    if args.name:
        sett.NAME = args.name
    if args.debug or sett.GENERAL.DEBUG:
        sett.read_confs('train_debug')
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
    if not sett.PATH.CPM_DATASET:
        sett.PATH.CPM_DATASET = DEFAULT_PATH.CPM_DATASET

    return sett


def main():
    sett = get_settings()

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
    elif os.path.exists(experiment_root):
        run_id = generate_run_id()
        sett.NAME = sett.NAME + '-' + run_id
        experiment_root = os.path.join(sett.PATH.EXPERIMENT_ROOT, sett.NAME)
    os.makedirs(experiment_root, exist_ok=True)

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=sett.GENERAL.LOGGING,
        handlers=[
            logging.FileHandler(os.path.join(experiment_root, 'train.log')),
            logging.StreamHandler()
            ]
        )

    # Load seeds
    random.seed(sett.GENERAL.SEED)
    np.random.seed(sett.GENERAL.SEED)
    torch.manual_seed(sett.GENERAL.SEED)

    logger.info("Start Training.")
    logger.info("Setting:" + str(sett))
    sett.get_toml_dict(os.path.join(experiment_root, 'train.toml'))

    train_dataset = WormsDataset(
        sett.PATH.WORMS_DATASET,
        sett.PATH.CPM_DATASET,
        patch_size=sett.DATA.PATCH_SIZE,
        n_consistent_worms=sett.DATA.N_CONSISTENT_WORMS,
        use_leftout_labels=sett.DATA.USE_LEFTOUT_LABELS,
        use_coord=sett.DATA.USE_COORD,
        normalize=sett.DATA.NORMALIZE,
        augmentation=sett.TRAIN.AUGMENTATION)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False,
                                               num_workers=sett.DATA.N_WORKER)

    model = PixelwiseModel(sett.MODEL.MODEL_NAME,
                           sett.MODEL.MODEL_PARAMS,
                           load_model_path=load_model_path)
    mock_input = next(iter(train_loader))
    model.print_model_summary(mock_input['raw'])

    tb_logs_dir = os.path.join(experiment_root, 'tblogs')
    os.makedirs(tb_logs_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_logs_dir)

    model_save_path = os.path.join(experiment_root, 'ckpts')
    os.makedirs(model_save_path, exist_ok=True)
    model.fit(train_loader, sett.TRAIN.N_STEP, sett.TRAIN.LEARNING_RATE, sett.TRAIN.WEIGHT_DECAY, sett.TRAIN.LR_DROP_FACTOR,
              sett.TRAIN.LR_DROP_PATIENCE, sett.TRAIN.MODEL_CKPT_EVERY_N_STEP, model_save_path,
              sett.TRAIN.RUNNING_LOSS_INTERVAL, sett.TRAIN.BURN_IN_STEP, tb_writer)

    tb_writer.close()
    logger.info('Finished Training!!!')


if __name__ == '__main__':
    main()
