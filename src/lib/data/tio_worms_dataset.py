'''dataset with torchio to handle splitting and stitching, augmentations, and normalizations.
TODO: add use_coord to dataset implementation'''

import glob
import logging
import pickle
import h5py
import os
import random
import numpy as np
from itertools import islice, combinations

from torch.utils.data import IterableDataset, Dataset, DataLoader
import torch
import torchio as tio

from scripts.settings import Settings

logger = logging.getLogger(__name__)


def get_transforms_from_sett(sett: Settings, train :bool):
    tr_trsfm_sett = sett.DATA.TRAIN_TRANSFORMS
    common_trsfm_sett = sett.DATA.COMMON_TRANSFORMS

    trsfm_list = []

    if train:
        if tr_trsfm_sett.RANDOM_FLIP > 0:
            trsfm_list.append(tio.RandomFlip(axes=(0,1,2), flip_probability=tr_trsfm_sett.RANDOM_FLIP))
        if tr_trsfm_sett.RANDOM_AFFINE.P > 0:
            ra_sett = tr_trsfm_sett.RANDOM_AFFINE
            trsfm_list.append(
                tio.RandomAffine(
                    scales=ra_sett.SCALE,
                    translation=ra_sett.TRANSLATION,
                    degrees=ra_sett.DEGREE,
                    p=ra_sett.P
                    )
                )
        if tr_trsfm_sett.RANDOM_ELASTIC.P > 0:
            re_sett = tr_trsfm_sett.RANDOM_ELASTIC
            trsfm_list.append(
                tio.RandomElasticDeformation(
                    num_control_points=re_sett.NUM_CONTROL_POINTS,
                    max_displacement=re_sett.MAX_DISPLACEMENT,
                    p=re_sett.P
                    )
                )
    # common transforms, rescale in the end probably better
    if common_trsfm_sett.RESCALEINTENSITY01:
        trsfm_list.append(tio.RescaleIntensity(out_min_max=(0,1)))
    return tio.Compose(trsfm_list)


def relabel_subject(subject, relabel_map):
    tmp_seghyp = torch.zeros_like(subject['seghyp'][tio.DATA])
    for k, v in relabel_map.items():
        locations = subject['seghyp'][tio.DATA] == k
        tmp_seghyp[locations] = v
    subject['seghyp'][tio.DATA] = tmp_seghyp
    return subject


def relabel_patches(patches, cpm_data):
    """returns a list of relabeled patches + a valid push force boolean matrix"""
    unique_patch_labels = [np.unique(patch['seghyp'][tio.DATA]) for patch in patches]
    wuids = [patch['wuid'] for patch in patches]

    if len(wuids) == 1:
        relabel_map = {k:v for v, k in enumerate(unique_patch_labels[0])}
        # get rid of background relabeling
        relabel_map.pop(0)
        return [relabel_subject(sub, relabel_map) for sub in patches], None
    elif len(wuids) == 2:
        # find the correct order i.e wuid0<wuid1
        compact_list_to_sort = zip([int(wuid) for wuid in wuids], unique_patch_labels, range(len(wuids)))
        compact_list_to_sort = sorted(compact_list_to_sort)
        wuids, unique_patch_labels, reorder_idxs = zip(*compact_list_to_sort)

        cpm = cpm_data[f'{wuids[0]}-{wuids[1]}']

        relabel_map21 = {k:v for v, k in enumerate(unique_patch_labels[1])}
        unique_target_labels = list(relabel_map21.values())
        assert relabel_map21[0] == 0, 'sth is wrong'
        relabel_map21.pop(0)
        curr_relabel_id = max(unique_target_labels) + 1
        relabel_map12 = {}
        for l1 in unique_patch_labels[0]:
            if l1 == 0:
                continue
            if l1 not in cpm.keys():
                relabel_map12[l1] = curr_relabel_id
                curr_relabel_id += 1
            else:  # if it is in the cpm data, we might have already assigned a label to it, or it is not present in the patch
                l2 = cpm[l1]
                if l2 in relabel_map21.keys():
                    relabel_map12[l1] = relabel_map21[l2]
                else:
                    relabel_map12[l1] = curr_relabel_id
                    curr_relabel_id += 1
            # unique_target_labels = list(set(unique_target_labels + list(relabel_map12.values())))
            # print('max unique_target_label is:', max(unique_target_labels))
            # print('len of it:', len(unique_target_labels))
        unique_target_labels = list(set(unique_target_labels + list(relabel_map12.values())))

        # now construct the valid_push_force_assignments
        vpf_size = len(unique_target_labels)
        valid_push_forces = np.ones([vpf_size, vpf_size], dtype=np.bool)
        # find all target labels that were not in the cpm data, there shouldn't be any push forces between them,
        # but pushed away from the rest and the rest from each other too
        invalidpf_target_labels = [v for k, v in relabel_map12.items() if k not in cpm.keys()]
        invalidpf_target_labels.extend([v for k, v in relabel_map21.items() if k not in cpm.values()])
        invalidpf_target_labels = set(invalidpf_target_labels)
        for i,j in combinations(invalidpf_target_labels, 2):
            valid_push_forces[i, j] = 0
            valid_push_forces[j, i] = 0

        _, correctly_ordered_relabel_map = zip(*sorted(zip(reorder_idxs, [relabel_map12, relabel_map21])))
        relabeled_patches = [relabel_subject(sub, relabel_map) for sub, relabel_map in zip(patches, correctly_ordered_relabel_map)]
        return relabeled_patches, valid_push_forces

    else:
        raise NotImplementedError()


class TrainTioWormsDataset(IterableDataset):
    def __init__(self,
                 dataset_root,
                 sett: Settings,
                 transforms=None,
                 debug=False):
        super(TrainTioWormsDataset).__init__()
        # list hdf files, either from files.txt or by listing all hdf files in a directory, or by a python list of files
        if os.path.isfile(dataset_root) and dataset_root.endswith('.txt'):
            with open(dataset_root, 'r') as f:
                self.hdf_fnlist = [fn for fn in f]
        elif isinstance(dataset_root, list):
            self.hdf_fnlist = []
            for fn in dataset_root:
                assert os.path.isfile(fn), f'cannot find file: {fn}'
                self.hdf_fnlist.append(fn)
        else:
            self.hdf_fnlist = glob.glob(os.path.join(dataset_root, "*.hdf"))
        logger.debug(f'Train Dataset hdf list:{self.hdf_fnlist}')

        # cpm data
        cpm_dataset = sett.DATA.CPM_DATASET
        assert cpm_dataset or sett.DATA.N_CONSISTENT_WORMS ==1, 'cpm_dataset has to be provided for ' \
            'training datasets with n_consistent_worms higher than 1'
        self.cpm_data = None
        if cpm_dataset:
            with open(cpm_dataset, 'rb') as f:
                self.cpm_data = pickle.load(f)

        self.debug = debug
        # Get dataset params from sett
        self.input_size = sett.DATA.INPUT_SIZE
        self.output_size = sett.DATA.OUTPUT_SIZE
        self.n_consistent_worms = sett.DATA.N_CONSISTENT_WORMS
        self.max_ninstance = sett.DATA.MAX_NINSTANCE
        self.min_label_volume = sett.DATA.MIN_LABEL_VOLUME
        self.samples_per_volume = sett.DATA.SAMPLES_PER_VOLUME_FOR_TRAINING
        if self.n_consistent_worms > 2:
            raise NotImplementedError()

        # simple settings, sampler could be better here
        self.transforms = transforms if transforms else get_transforms_from_sett(sett, train=True)
        self.sampler = tio.data.UniformSampler(patch_size=self.input_size)

    def __iter__(self):
        while True:
            sample = {}
            fn_list = random.sample(set(self.hdf_fnlist), self.n_consistent_worms)
            subjects = []
            for fn in fn_list:
                with h5py.File(fn, 'r') as f:
                    raw = f['raw'][()]
                    raw = np.expand_dims(raw, 0)
                    seghyp = f['gt_seghyp'][()]  # dtype: uint16
                    seghyp = np.expand_dims(seghyp, 0)
                subject = tio.Subject(
                    raw=tio.ScalarImage(tensor=raw),
                    seghyp=tio.LabelMap(tensor=seghyp),  # dtype: int32
                    wuid=fn.split('.hdf')[0][-2:]
                    )
                subjects.append(subject)
            if self.debug:
                sample['original_subjects'] = tio.SubjectsDataset(subjects)
            subjects = tio.SubjectsDataset(subjects, transform=self.transforms)  # normalization and augmentation here
            if self.debug:
                sample['transformed_subjects'] = subjects
            sampler_iterator_list = [self.sampler(subject) for subject in subjects]
            for _ in range(self.samples_per_volume):
                patches = [next(iter) for iter in sampler_iterator_list]
                # For some reason, sometimes (when elastic happens), seghyp dtype changes to float which
                # might cause issues with unique call functions or sets
                for patch in patches:
                    patch['seghyp'][tio.DATA] = patch['seghyp'][tio.DATA].to(torch.int)
                # crop output patches to output size
                cropsize = [(ai - bi) // 2 for ai, bi in zip(self.input_size, self.output_size)]
                patches = [tio.Crop(cropsize, exclude=['raw', 'wuid'])(patch) for patch in patches]
                if self.debug:
                    sample['patches_before_relabeling'] = patches
                patches, valid_discloss_pushforce = relabel_patches(patches, self.cpm_data)
                if self.debug:
                    sample['relabeled_patches'] = patches

                # expand seghyps into their own dimenstion to make it work easier with discloss implementation
                # Disc loss especially for very large n_instance cases, throws GPU OOM Error. While there has been
                # some improvements for Disc loss, it still remains an issue for larger n_instance sizes. E.g. for
                # patch size of [64, 64, 64], it throws OOM error for n_instance 140. However, it does not for 120.
                # Still havn't done an extensive study of the exact n_instance number, but use max_ninstance
                # parameter to randomely select max_ninstance number if it has more.

                # get max number of labels
                unique_labels = []
                for patch in patches:
                    unique_labels.extend(np.unique(patch['seghyp'][tio.DATA]))
                unique_labels = list(np.unique(unique_labels))
                max_l = len(unique_labels)
                if max_l < 3:
                    continue  # doesnt make sense to have only background and one label

                patches = [tio.transforms.OneHot(num_classes=max_l)(patch) for patch in patches]
                seghyps = [patch['seghyp'][tio.DATA] for patch in patches]

                # select some of the labels
                selected_rows = set(unique_labels)
                # dont select those labels that are very small
                num_pixel_per_label = torch.stack([seghyp.sum([1,2,3]) for seghyp in seghyps]).sum([0])
                small_labels_to_remove = torch.nonzero(num_pixel_per_label<self.min_label_volume).squeeze().tolist()
                if isinstance(small_labels_to_remove, int):
                    small_labels_to_remove = [small_labels_to_remove]
                selected_rows -= set(small_labels_to_remove)
                if self.max_ninstance and self.max_ninstance != 0 and len(selected_rows) > self.max_ninstance:
                    logger.debug(
                        f'Clipping input seghyp data from original:[{len(selected_rows)}] n_instances to [{self.max_ninstance}]')
                    selected_rows = random.sample(selected_rows-set([0]), self.max_ninstance-1)
                    selected_rows.append(0)
                    # include background and make sure it is sorted
                selected_rows = sorted(selected_rows)

                # now in the end, just choose the selected_rows of seghyp and valid_push_force

                # in the end, select selected rows, and make sure they are torch.tensors to be able to use pin_memory
                valid_discloss_pushforce = valid_discloss_pushforce[np.ix_(selected_rows, selected_rows)]
                sample['raw'] = torch.cat(
                    [patch['raw'][tio.DATA] for patch in patches],
                    dim=0
                    )
                seghyps = [seghyp[selected_rows, ...] for seghyp in seghyps]
                sample['seghyp'] = torch.cat(
                    [torch.unsqueeze(seghyp, 0) for seghyp in seghyps],
                    dim=0
                    )
                sample['valid_discloss_pushforce'] = torch.tensor(valid_discloss_pushforce)
                yield sample
                continue

                # Add coord channels to raw data based on corner and patchsize
                # if self.use_coord:
                #     xyz_coords = np.meshgrid(np.arange(start=corner_input[0], stop=corner_input[0]+self.patch_size[0]),
                #                              np.arange(start=corner_input[1], stop=corner_input[1]+self.patch_size[1]),
                #                              np.arange(start=corner_input[2], stop=corner_input[2]+self.patch_size[2]))
                #     xyz_coords = [np.expand_dims(c, axis=0) for c in xyz_coords]
                #     xyz_coords = np.vstack(xyz_coords)
                #     for i, raw in enumerate(raws):
                #         raws[i] = np.vstack([raw, xyz_coords])


class TestTioWormsDataset(IterableDataset):
    '''Just loads the data from hdf files and makes sure that the same transformations are applied to input as in
    train except for augmentations. they are whole images and no cropping is done (can't be put directly as input to
    the network.)
    '''
    def __init__(self,
                 dataset_root,
                 sett: Settings,
                 transforms=None,
                 debug=False):
        super(TrainTioWormsDataset).__init__()
        # list hdf files, either from files.txt or by listing all hdf files in a directory, or by a python list of files
        if os.path.isfile(dataset_root) and dataset_root.endswith('.txt'):
            with open(dataset_root, 'r') as f:
                self.hdf_fnlist = [fn for fn in f]
        elif isinstance(dataset_root, list):
            self.hdf_fnlist = []
            for fn in dataset_root:
                assert os.path.isfile(fn), f'cannot find file: {fn}'
                self.hdf_fnlist.append(fn)
        else:
            self.hdf_fnlist = glob.glob(os.path.join(dataset_root, "*.hdf"))
        logger.debug(f'Test Dataset hdf list:{self.hdf_fnlist}')