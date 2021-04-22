import argparse
import logging
import re

import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import _init_paths

from lib.data.worms_dataset import WormsDatasetOverSeghypCenters
from lib.models.pixelwise_model import PixelwiseModel
from settings import Settings, DefaultPath

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # config is mainly needed for model instantiation
    parser.add_argument('-c', '--config', required=True, action='append')
    parser.add_argument('-m', '--model', action='append',
                        help='path to .pth saved model, to compute cluster centers from. If multiple path given, '
                             'runs it for all of them. If experiment root is given, searches for all .pth models in '
                             'path/ckpts. either in absolute path or relative to default experiments directory. If '
                             'given, overrides values from config i.e. experiment_root/name')
    parser.add_argument('--only_best', action='store_true',
                        help='onyl uses models starting with bestmodel* for clustering')
    args = parser.parse_args()
    return args


def main(args):
    # cluster settings
    sett = Settings(args.config)
    DEFAULT_PATH = DefaultPath()
    if not sett.PATH.EXPERIMENT_ROOT:
        sett.PATH.EXPERIMENT_ROOT = DEFAULT_PATH.EXPERIMENTS
    if not sett.PATH.WORMS_DATASET:
        sett.PATH.WORMS_DATASET = DEFAULT_PATH.WORMS_DATASET
    if not sett.PATH.CPM_DATASET:
        sett.PATH.CPM_DATASET = DEFAULT_PATH.CPM_DATASET

    experiment_root = os.path.join(sett.PATH.EXPERIMENT_ROOT, sett.NAME)

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=10 if sett.GENERAL.DEBUG else 20,
        handlers=[
            logging.FileHandler(os.path.join(experiment_root, 'cluster.log')),
            logging.StreamHandler()
            ]
        )

    # Load seeds
    if sett.GENERAL.SEED:
        random.seed(sett.GENERAL.SEED)
        np.random.seed(sett.GENERAL.SEED)
        torch.manual_seed(sett.GENERAL.SEED)


    model_pth_list = []
    # if args.model not given, read from config
    if not args.model:
        args.model = [experiment_root]
    for m in args.model:
        if m.endswith('.pth'):
            if not os.path.isfile(m):
                model_pth_list.append(os.path.join(experiment_root, 'ckpts', m))
        else:
            # infer ckpt root
            ckpt_root = None
            if os.path.exists(os.path.join(m, 'ckpts')):
                ckpt_root = os.path.join(m, 'ckpts')
            elif os.path.exists(os.path.join(DEFAULT_PATH.EXPERIMENTS, m, 'ckpts')):
                ckpt_root = os.path.join(DEFAULT_PATH.EXPERIMENTS, m, 'ckpts')
            assert ckpt_root, f'Dont know what to do with this [{m}]'
            model_pth_list.extend([os.path.join(ckpt_root, f) for f in os.listdir(ckpt_root) if f.endswith('.pth')])

    if args.only_best:
        model_pth_list_tmp = [f for f in model_pth_list if f.split('/')[-1].startswith('bestmodel')]
        model_pth_list = model_pth_list_tmp

    # First sort them to not get the zigzag plots in tensorboard
    p = re.compile(r'.+-step=(\d+).+')
    model_step_list = [int(p.search(m.split('/')[-1]).group(1)) for m in model_pth_list]
    model_step_list, model_pth_list = zip(*sorted(zip(model_step_list, model_pth_list)))

    logger.info(f'Clustering for [{len(model_pth_list)}] models:')
    for f in model_pth_list:
        logger.info(f'--- {f}')

    tb_logs_dir = os.path.join(experiment_root, 'tblogs')
    os.makedirs(tb_logs_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_logs_dir)

    # ==========================
    # Now run clustering for every model in model_pth_list and save clustering centers next to .pth model with
    # .cluster.joblib
    test_loader = WormsDatasetOverSeghypCenters(
        sett.PATH.WORMS_DATASET,
        patch_size=sett.DATA.PATCH_SIZE,
        output_size=sett.DATA.OUTPUT_SIZE,
        use_coord=sett.DATA.USE_COORD,
        normalize=sett.DATA.NORMALIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=1)
    for i, model_pth in enumerate(model_pth_list):
        model_name = '.'.join(model_pth.split('/')[-1].split('.')[:-1])
        model_path = os.path.join(*model_pth.split('/')[:-1])
        step = model_step_list[i]
        cluster_save_file = os.path.join('/', model_path, model_name + '.cluster.joblib')

        model = PixelwiseModel(sett.MODEL.MODEL_NAME, sett.MODEL.MODEL_PARAMS,
                               padding=sett.MODEL.PADDING,
                               load_model_path=model_pth)
        logger.info(f'Start Clustering. n_cluster=[{sett.TRAIN.N_CLUSTER}]')
        save_embedding_image_file_path = os.path.join(experiment_root, 'output', 'plots',
                                                      model_name+'-cluster_embedding')
        cluster_centers = model.compute_cluster_centers(sett.TRAIN.N_CLUSTER, test_loader, cluster_save_file,
                                                        num_workers=sett.DATA.N_WORKER, tb_writer=tb_writer,
                                                        save_embedding_image_file_path=save_embedding_image_file_path)


if __name__ == '__main__':
    args = get_args()
    main(args)