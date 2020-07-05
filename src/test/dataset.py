from lib.data.siamese_worms_dataset import SiameseWormsDataset
from lib.utils.general import get_config

import _init_paths


if __name__ == '__main__':
    print('testing...')
    config_file = '/home/ashkan/workspace/deployed/worms_nuclei_metric_learning-deployed/experiments_cfg/default.toml'
    conf = get_config([config_file])

    dataset = SiameseWormsDataset(conf['path']['worms_dataset'],
                                  conf['path']['cpm_dataset'],
                                  patch_size=(100,100,100))
    sample = iter(dataset).__next__()
    print("Finished!!!")