from lib.data.worms_dataset import WormsDataset
from lib.utils.general import get_config
from settings import Settings, DEFAULT_PATH

import _init_paths


if __name__ == '__main__':
    print('testing...')
    sett = Settings(['train_default', 'train_debug'])
    if not sett.PATH.EXPERIMENT_ROOT:
        sett.PATH.EXPERIMENT_ROOT = DEFAULT_PATH.EXPERIMENTS
    if not sett.PATH.WORMS_DATASET:
        sett.PATH.WORMS_DATASET = DEFAULT_PATH.WORMS_DATASET
    if not sett.PATH.CPM_DATASET:
        sett.PATH.CPM_DATASET = DEFAULT_PATH.CPM_DATASET

    # train mode
    sett.DATA.N_CONSISTENT_WORMS = 2
    dataset = WormsDataset(sett.PATH.WORMS_DATASET,
                                  sett.PATH.CPM_DATASET,
                                  patch_size=sett.DATA.PATCH_SIZE,
                                  n_consistent_worms=sett.DATA.N_CONSISTENT_WORMS,
                           use_leftout_labels=sett.DATA.USE_LEFTOUT_LABELS,
                           use_coord=sett.DATA.USE_COORD,
                           normalize=sett.DATA.NORMALIZE,
                           augmentation=sett.TRAIN.AUGMENTATION,
                           debug=True)
    di = iter(dataset)
    for _ in range(10):
        s = next(di)
    print("Finished!!!")