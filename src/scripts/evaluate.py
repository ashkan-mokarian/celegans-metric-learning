import argparse
import logging
import re

import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import plotly.offline as plotlyoff
import plotly.graph_objs as plotlygo

import _init_paths

# from data.siamese_worms_dataset import SiameseWormsDataset
from lib.data.worms_dataset import WormsDatasetOverSeghypCenters
from lib.models.pixelwise_model import PixelwiseModel
from settings import Settings, DefaultPath

# TODO: this is here awkward
from consolidate_cpm_dataset import helper_get_pm_and_acc

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
                        help='only uses models starting with bestmodel* for clustering')
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
        level=sett.GENERAL.LOGGING,
        handlers=[
            logging.FileHandler(os.path.join(experiment_root, 'evaluation.log')),
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

    logger.info(f'Evaluating for [{len(model_pth_list)}] models:')
    for f in model_pth_list:
        logger.info(f'--- {f}')

    tb_logs_dir = os.path.join(experiment_root, 'tblogs')
    os.makedirs(tb_logs_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_logs_dir)

    # ==========================
    # Now run prediction for every model in model_pth_list and save clustering results
    test_loader = WormsDatasetOverSeghypCenters(sett.PATH.WORMS_DATASET, patch_size=sett.DATA.PATCH_SIZE,
                                                output_size=sett.DATA.OUTPUT_SIZE,
                                                use_coord=sett.DATA.USE_COORD, normalize=sett.DATA.NORMALIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=1)

    pairwise_acc_per_step_dict = {}
    pairwise_acc_plot_file = os.path.join(experiment_root, 'output', 'plots', 'pairwise_accuracy_boxplot.html')

    for model_pth in model_pth_list:
        model_name = '.'.join(model_pth.split('/')[-1].split('.')[:-1])
        model_path = os.path.join(*model_pth.split('/')[:-1])
        step = int(model_name.split('step=')[-1].split('-')[0].split('.')[0])
        cluster_load_file = os.path.join('/', model_path, model_name + '.cluster.joblib')
        evaluate_save_file = os.path.join('/', model_path, '..', 'output', model_name + '-cluster_predict_per_worm.pkl')
        os.makedirs(os.path.dirname(evaluate_save_file), exist_ok=True)

        model = PixelwiseModel(sett.MODEL.MODEL_NAME, sett.MODEL.MODEL_PARAMS,
                               padding=sett.MODEL.PADDING,
                               load_model_path=model_pth)
        logger.info(f'Start Evaluating for model at step:[{step}] - path:[{model_pth}]')

        # iterate over loader, one worm at a time, because predict works on single wormoverseghypdataset
        allworm_sl_2_cl_dict = {}
        allworm_sl_2_gl_dict = {}
        for n_worm, worm_dataset in enumerate(test_loader):
            logger.info(f'worm: {n_worm+1}/{len(test_loader)} - n_seghyp={len(worm_dataset)} - worm_fn='
                        f'{worm_dataset.worm_data}')

            worm_fn = worm_dataset.worm_data
            wormuid = int(worm_fn.split('/')[-1].split('.')[0].split('worm')[-1])
            sl_2_cl, sl_2_gl = model.predict(oneworm_dataset_over_seghypcenters=worm_dataset,
                          cluster_load_file=cluster_load_file,
                          num_workers=sett.DATA.N_WORKER)
            allworm_sl_2_cl_dict.update({wormuid:sl_2_cl})
            allworm_sl_2_gl_dict.update({wormuid:sl_2_gl})

        saved_pickle = {'allworm_seghyplabel_to_clusterlabel_dict': allworm_sl_2_cl_dict,
                        'allworm_seghyplabel_to_gtlabel_dict': allworm_sl_2_gl_dict,
                        'model_name': model_name,
                        'model_fn': model_pth,
                        'step': step}
        with open(evaluate_save_file, 'wb') as f:
            pickle.dump(saved_pickle, f)

        # some evaluation metrics for clustering using sklearn.metrics
        labels_true = []
        labels_pred = []
        # leave the ones with gt 0 labels
        for wuid in allworm_sl_2_cl_dict.keys():
            sl_2_cl_dict = allworm_sl_2_cl_dict[wuid]
            sl_2_gl_dict = allworm_sl_2_gl_dict[wuid]
            for sl, cl in sl_2_cl_dict.items():
                gt_label = sl_2_gl_dict[sl]
                if gt_label != 0:
                    labels_pred.append(cl)
                    labels_true.append(gt_label)
        # now compute bunch of metrics based on the two lists
        ari = metrics.adjusted_rand_score(labels_true, labels_pred)
        tb_writer.add_scalar('evaluation/adjusted_rand_index', ari, step)

        nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
        tb_writer.add_scalar('evaluation/normalized_mutual_information', nmi, step)

        homogenity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
        tb_writer.add_scalar('evaluation/homogenity', homogenity, step)
        tb_writer.add_scalar('evaluation/completeness', completeness, step)
        tb_writer.add_scalar('evaluation/v_measure', v_measure, step)

        fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels_true, labels_pred)
        tb_writer.add_scalar('evaluation/fowlkes_mallows_score', fowlkes_mallows_score, step)

        logger.info(f"Evaluation Metrics:\n"
                    f"===================\n"
                    f"adjusted rand index:   {ari}\n"
                    f"normalized mutual inf: {nmi}\n"
                    f"homogenity:            {homogenity}\n"
                    f"completeness:          {completeness}\n"
                    f"v_measure:             {v_measure}\n"
                    f"fowlkes_mallows_score: {fowlkes_mallows_score}")

        # Compute pairwise matching accuracy for every two worms, and avg, and plot sth
        allworm_sl_2_gl_dict = saved_pickle['allworm_seghyplabel_to_gtlabel_dict']
        allworm_sl_2_cl_dict = saved_pickle['allworm_seghyplabel_to_clusterlabel_dict']
        step = saved_pickle['step']

        pairwise_acc_per_step_dict.update({step:[]})
        n_worms = len(allworm_sl_2_cl_dict.keys())
        for i in range(1, n_worms):
            for j in range(i+1, n_worms+1):
                sl_2_cl_i = allworm_sl_2_cl_dict[i]
                sl_2_cl_j = allworm_sl_2_cl_dict[j]
                cl_2_sl_j = {v:k for k, v in sl_2_cl_j.items()}
                sl_2_gl_i = allworm_sl_2_gl_dict[i]
                sl_2_gl_j = allworm_sl_2_gl_dict[j]

                # compute_accuracy
                w1_w2_pm = {}
                for w1sl, w1cl in sl_2_cl_i.items():
                    if w1cl not in cl_2_sl_j.keys():
                        continue
                    w1_w2_pm.update({w1sl: cl_2_sl_j[w1cl]})
                acc = len([1 for k, v in w1_w2_pm.items() if sl_2_gl_i[k]==sl_2_gl_j[v]])
                logger.info(f'#correct_matches: step=[{step}] - worms=[{i:02},{j:02}] = {acc}')
                pairwise_acc_per_step_dict[step].append(acc)

    # this is bad implementation, but write acc results for lisa original data as step 0
    N_WORMS = 30
    pairwise_acc_per_step_dict.update({0:[]})
    for i in range(1, N_WORMS):
        for j in range(i+1, N_WORMS+1):
            w1_w2_pm, acc = helper_get_pm_and_acc(i, j)
            pairwise_acc_per_step_dict[0].append(acc)
            # since there are two directions
            w2_w1_pm, acc = helper_get_pm_and_acc(j, i)
            pairwise_acc_per_step_dict[0].append(acc)
    # ===========================
    fig = plotlygo.Figure()
    keys = sorted(list(pairwise_acc_per_step_dict.keys()))
    for k in keys:
        fig.add_trace(plotlygo.Box(
            y=pairwise_acc_per_step_dict[k],
            name=k
        ))
    plotlyoff.plot(fig, filename=pairwise_acc_plot_file, auto_open=False)


if __name__ == '__main__':
    args = get_args()
    main(args)