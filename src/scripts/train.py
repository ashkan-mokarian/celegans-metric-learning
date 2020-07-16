import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from settings import Settings
from lib.utils.general import generate_run_id
from lib.data.siamese_worms_dataset import SiameseWormsDataset
from lib.models.siamese_pixelwise_model import SiamesePixelwiseModel

logger = logging.getLogger(__name__)

# Some constants for now
PATCH_SIZE = (52, 52, 52)


def get_settings():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True, action='append',
                        help='path to config file, or name of configs to load sequentially, if not path and only ' \
                             'config name, assumes default dir structure')
    parser.add_argument('-n', '--name', type=str,
                        help='name of training session, used for some namings')
    parser.add_argument('--debug', action='store_true',
                        help='loads train_debug.toml for debug train mode')
    args = parser.parse_args()

    # overwrite .toml settings with CLI settings
    ts = Settings(args.config)
    if args.name:
        ts.NAME = args.name
    if args.debug or ts.GENERAL.DEBUG:
        ts.read_confs('train_debug')
    return ts


def main():
    ts = get_settings()

    # experiment root
    experiment_root = os.path.join(ts.PATH.EXPERIMENTS, ts.NAME)
    load_model_path = None
    if ts.MODEL.INIT_MODEL_BEST:
        raise NotImplementedError
    elif ts.MODEL.INIT_MODEL_PATH:
        raise NotImplementedError
    elif ts.MODEL.INIT_MODEL_LAST:
        ckpts_root = os.path.join(experiment_root, 'ckpts')
        ckpts = [os.path.join(ckpts_root, f) for f in os.listdir(ckpts_root) if f.endswith('.pth')]
        load_model_path = max(ckpts, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        assert os.path.isfile(load_model_path)
    elif os.path.exists(experiment_root):
        run_id = generate_run_id()
        ts.NAME = ts.NAME + '-' + run_id
        experiment_root = os.path.join(ts.PATH.EXPERIMENTS, ts.NAME)
    os.makedirs(experiment_root, exist_ok=True)

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=ts.GENERAL.LOGGING,
        handlers=[
            logging.FileHandler(os.path.join(experiment_root, 'train_log.txt')),
            logging.StreamHandler()
            ]
        )



    # Load seeds
    random.seed(ts.GENERAL.SEED)
    np.random.seed(ts.GENERAL.SEED)
    torch.manual_seed(ts.GENERAL.SEED)

    logger.info("Start Training.")
    logger.info("Setting:" + str(ts))

    train_dataset = SiameseWormsDataset(ts.PATH.WORMS_DATASET, ts.PATH.CPM_DATASET,
                                        patch_size=PATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, num_workers=1)

    model = SiamesePixelwiseModel(ts.MODEL.MODEL_NAME,
                                  ts.MODEL.MODEL_PARAMS,
                                  load_model_path=load_model_path)
    mock_input = iter(train_loader).__next__()
    model.print_model_summary(mock_input)

    tb_logs_dir = os.path.join(experiment_root, 'tblogs')
    os.makedirs(tb_logs_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_logs_dir)

    model_save_path = os.path.join(experiment_root, 'ckpts')
    os.makedirs(model_save_path, exist_ok=True)
    model.fit(train_loader, ts.TRAIN.N_STEP, ts.TRAIN.LEARNING_RATE, ts.TRAIN.WEIGHT_DECAY, ts.TRAIN.LR_DROP_FACTOR,
              ts.TRAIN.LR_DROP_PATIENCE, 10, model_save_path, tb_writer)

    tb_writer.close()
    logger.info('Finished Training!!!')


if __name__ == '__main__':
    main()