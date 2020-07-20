import glob
import logging
import pickle

import h5py
import os
import random
import numpy as np

from torch.utils.data import IterableDataset, Dataset
import torch

logging.getLogger(__name__)

# TODO: needs work for train=False. 1. one pass through all samples and not forever. 2. Since I am clustering at end
#  all of the results, should do it somehow memory efficient.
class SiameseWormsDataset(IterableDataset):
    """Siamese implementation of matching labels worms dataset"""
    def __init__(self,
                 worms_dataset_root,
                 cpm_dataset,
                 patch_size,
                 train=True):
        super(SiameseWormsDataset).__init__()
        self.worms_data = glob.glob(os.path.join(worms_dataset_root, "*.hdf"))
        with open(cpm_dataset, 'rb') as f:
            self.cpm_data = pickle.load(f)
        self.train = train
        self.patch_size = patch_size
        # Calculate the valid range of lower left 3d corner to sample for a given patch size
        self.MAX_SAMPLE_X = 140 - self.patch_size[0]
        self.MAX_SAMPLE_Y = 140 - self.patch_size[1]
        self.MAX_SAMPLE_Z = 1166 - self.patch_size[2]

    def __iter__(self):
        """Sample the two worms, fetch data, relabel them according to cpm, do Transforms"""
        if self.train:
            while True:
                random2worms = random.sample(self.worms_data, k=2)
                w1_fn, w2_fn = random2worms[0], random2worms[1]
                w1id = int(w1_fn.split('/')[-1].split('.')[0])
                w2id = int(w2_fn.split('/')[-1].split('.')[0])
                # swap, bcuz cpm assumes w1<w2
                if w2id<w1id:
                    w1id, w2id = w2id, w1id
                    w1_fn, w2_fn = w2_fn, w1_fn
                cpm_12 = self.cpm_data[f'{w1id}-{w2id}']

                # sample patch corner from valid points
                corner = (random.randint(0, self.MAX_SAMPLE_X),
                          random.randint(0, self.MAX_SAMPLE_Y),
                          random.randint(0, self.MAX_SAMPLE_Z))
                sampled_patch = (slice(corner[0], corner[0]+self.patch_size[0]),
                                 slice(corner[1], corner[1]+self.patch_size[1]),
                                 slice(corner[2], corner[2]+self.patch_size[2]))

                # read data from the wormd_dataset .hdf files and extract sampled batch
                with h5py.File(w1_fn, 'r') as f:
                    raw1 = f['volumes/raw'][()][sampled_patch]
                    label1_orig = f['volumes/nuclei_seghyp'][()][sampled_patch]
                with h5py.File(w2_fn, 'r') as f:
                    raw2 = f['volumes/raw'][()][sampled_patch]
                    label2_orig = f['volumes/nuclei_seghyp'][()][sampled_patch]
                # relabel label1 and 2, dont leave empty labels, make sure they match according to cpm
                # Apparently instance segmentation datasets use a different axis for marking different instances.
                # This is also the way it is used in lib/losses/discriminative_loss.py
                # Therefore, instead of having an instance number for every pixel, the corresponding pixels go into their
                # corresponding n_instance axis, new shape is (n_instances,x,y,z)

                # Since I have to concatenate n_instances to create the 2 batch, take the max n_instances of the two labels
                nl = np.sum([1 if x in label1_orig or y in label2_orig else 0 for x, y in cpm_12.items()])
                # sometimes, when patch_size too small, no labels are found, discard those
                if nl == 0:
                    continue
                label1 = np.zeros((nl,) + label1_orig.shape)
                label2 = np.zeros((nl,) + label2_orig.shape)

                relabel_id = 0
                for l1, l2 in cpm_12.items():
                    increase = False
                    if l1 in label1_orig:
                        label1[relabel_id, label1_orig == l1] = 1
                        increase = True
                    if l2 in label2_orig:
                        label2[relabel_id, label2_orig == l2] = 1
                        increase = True
                    if increase:
                        relabel_id += 1

                # Add channel dimension to raw data
                raw1 = np.expand_dims(raw1, axis=0)
                raw2 = np.expand_dims(raw2, axis=0)

                nl = np.array(nl)
                # cast them to tensor objects
                sample = {'raw1': raw1,
                          'raw2': raw2,
                          'label1': label1,
                          'label2': label2,
                          'n_cluster': nl}
                sample = {k: torch.from_numpy(v.astype(np.int16)) for k, v in sample.items()}
                yield sample

        else:  # self.train == False
            raise NotImplemented


# TODO: This looks super STUPID. but for now
class WormsDatasetOverSeghypCenters:
    def __init__(self,
                 worms_dataset_root,
                 patch_size):
        # super(WormsDatasetOverSeghypCenters, self).__init__()
        self.worms_data = glob.glob(os.path.join(worms_dataset_root, "*.hdf"))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.worms_data)

    def __getitem__(self, idx):
        return OneWormDatasetOverSeghypCenters(self.worms_data[idx], patch_size=self.patch_size)


class OneWormDatasetOverSeghypCenters(Dataset):
    def __init__(self,
                 worm_data,
                 patch_size):
        super(OneWormDatasetOverSeghypCenters, self).__init__()
        self.worm_data = worm_data
        self.aaa = 0
        self.patch_size = np.array(patch_size)
        with h5py.File(self.worm_data, 'r') as f:
            self.raw = f['volumes/raw'][()]
            self.seghyp = f['volumes/nuclei_seghyp'][()]
            self.con_seghyp = f['matrix/con_seghyp'][()]
            self.gt_label = f['volumes/gt_nuclei_labels'][()]

    def __len__(self):
        return self.con_seghyp.shape[0]-1  # first one is background label 0 with con [0, 0, 0]

    def __getitem__(self, tmp_idx):
        idx = tmp_idx + 1
        con_idx = np.round(self.con_seghyp[idx]).astype(dtype=np.int)
        corner = con_idx - self.patch_size//2
        corner[corner < 0] = 0
        patch = (slice(corner[0], corner[0]+self.patch_size[0]),
                 slice(corner[1], corner[1]+self.patch_size[1]),
                 slice(corner[2], corner[2]+self.patch_size[2]))

        raw = self.raw[patch]
        raw = np.expand_dims(raw, axis=0)  # add channel dim
        masktmp = self.seghyp[patch]
        mask = np.zeros_like(masktmp)
        mask[masktmp==idx] = 1
        gt_label = self.gt_label[patch]
        gt_label_ids = gt_label[mask==1]
        gt_label_id = gt_label_ids[0]
        assert np.all(gt_label_ids == gt_label_id)
        gt_label_id = np.array(gt_label_id)
        sample = {'patch_reference_corner': corner,
                  'raw': raw,
                  'mask': mask,
                  'gt_label_id': gt_label_id}
        sample = {k: torch.from_numpy(v.astype(np.int16)) for k, v in sample.items()}
        return sample
