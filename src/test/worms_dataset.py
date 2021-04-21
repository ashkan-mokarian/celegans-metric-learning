from lib.data.tio_worms_dataset import TrainTioWormsDataset, get_transforms_from_sett
from lib.utils.general import get_config
from settings import Settings, DEFAULT_PATH
import logging, os
import time
import torch
import random

import _init_paths


if __name__ == '__main__':
    print('testing...')
    sett = Settings(['train_default'])
    if not sett.PATH.EXPERIMENT_ROOT:
        sett.PATH.EXPERIMENT_ROOT = DEFAULT_PATH.EXPERIMENTS
    # set logger
    logging.basicConfig(
        format='%(name)s:%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=10 if sett.GENERAL.DEBUG else 20,
        handlers=[
            logging.StreamHandler()
            ]
        )

    # train mode
    sett.DATA.N_CONSISTENT_WORMS = 2
    sett.DATA.MAX_NINSTANCE = 40
    dataset = TrainTioWormsDataset(
        sett.PATH.WORMS_DATASET,
        sett=sett,
        debug=False)
    # ds = torch.utils.data.BufferedShuffleDataset(dataset, buffer_size=12)
    def init_fn(worker_id):
        random.seed(worker_id+time.time())

    dl = torch.utils.data.DataLoader(dataset, num_workers=10, worker_init_fn=init_fn, batch_size=None, pin_memory=True,
                                     persistent_workers=True)
    dli = iter(dl)
    for i in range(100):
        start = time.time()
        s = next(dli)
        print(i, ':', time.time()-start)
    print("Finished!!!")