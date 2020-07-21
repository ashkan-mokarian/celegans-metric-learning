import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

import _init_paths
# from data.siamese_worms_dataset import SiameseWormsDataset
from lib.data.siamese_worms_dataset import WormsDatasetOverSeghypCenters
from lib.models.siamese_pixelwise_model import SiamesePixelwiseModel
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
    ss = Settings(args.config)
    DEFAULT_PATH = DefaultPath()
    if not ss.PATH.EXPERIMENT_ROOT:
        ss.PATH.EXPERIMENT_ROOT = DEFAULT_PATH.EXPERIMENTS
    if not ss.PATH.WORMS_DATASET:
        ss.PATH.WORMS_DATASET = DEFAULT_PATH.WORMS_DATASET
    if not ss.PATH.CPM_DATASET:
        ss.PATH.CPM_DATASET = DEFAULT_PATH.CPM_DATASET

    experiment_root = os.path.join(ss.PATH.EXPERIMENT_ROOT, ss.NAME)

    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=ss.GENERAL.LOGGING,
        handlers=[
            logging.FileHandler(os.path.join(experiment_root, 'evaluation_log.txt')),
            logging.StreamHandler()
            ]
        )

    # Load seeds
    if ss.GENERAL.SEED:
        random.seed(ss.GENERAL.SEED)
        np.random.seed(ss.GENERAL.SEED)
        torch.manual_seed(ss.GENERAL.SEED)


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

    logger.info(f'Clustering for [{len(model_pth_list)}] models:')
    for f in model_pth_list:
        logger.info(f'--- {f}')

    # ==========================
    # Now run clustering for every model in model_pth_list and save clustering centers next to .pth model with
    # .cluster.npy
    test_loader = WormsDatasetOverSeghypCenters(ss.PATH.WORMS_DATASET, patch_size=ss.DATA.PATCH_SIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=1)
    tb_logs_dir = os.path.abspath(os.path.join('/', *model_pth_list[0].split('/')[:-1], '..', 'tblogs'))
    print(tb_logs_dir)
    tb_writer = SummaryWriter(log_dir=tb_logs_dir)
    for model_pth in model_pth_list:
        model_name = '.'.join(model_pth.split('/')[-1].split('.')[:-1])
        model_path = os.path.join(*model_pth.split('/')[:-1])
        step = int(model_name.split('step=')[-1].split('-')[0].split('.')[0])
        cluster_save_file = os.path.join('/', model_path, model_name + '.cluster.joblib')
        evaluate_save_file = os.path.join('/', model_path, '..', 'results', model_name + '-cluster_results.pkl')

        # os.makedirs(evaluate_save_file, exist_ok=True)

        model = SiamesePixelwiseModel(ss.MODEL.MODEL_NAME, ss.MODEL.MODEL_PARAMS,
                                  load_model_path=model_pth)
        logger.info(f'Start Evaluating. n_cluster=[{ss.TRAIN.N_CLUSTER}]')
        cluster_results = model.evaluate(test_loader, cluster_save_file,
                                                        num_workers=ss.DATA.N_WORKER, tb_writer=tb_writer)

        # pickle.dump(evaluate_save_file, cluster_results)


if __name__ == '__main__':
    args = get_args()
    main(args)