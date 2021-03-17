# disable automatic batching of data, since we are slightly changing the definition of batch for this data. so Every
# iteration of dataset already yields a batch of data.

import glob
import logging
import pickle

import h5py
import os
import random
import numpy as np

from torch.utils.data import IterableDataset, Dataset
import torch
import augment

from lib.data.elastic_augment import create_elastic_transformation


logger = logging.getLogger(__name__)


class WormsDataset(IterableDataset):
    def __init__(self,
                 worms_dataset_root,
                 cpm_dataset,
                 n_consistent_worms,
                 use_leftout_labels,
                 use_coord,
                 normalize,
                 patch_size,
                 output_size,
                 augmentation=None,
                 transforms=None,
                 train=True,
                 debug=False,
                 max_ninstance=None):
        super(WormsDataset).__init__()

        self.worms_data = glob.glob(os.path.join(worms_dataset_root, "*.hdf"))
        with open(cpm_dataset, 'rb') as f:
            self.cpm_data = pickle.load(f)
        self.train = train
        self.patch_size = np.array(patch_size)
        self.output_size = np.array(output_size)

        self.n_consistent_worms = n_consistent_worms
        self.use_leftout_labels = use_leftout_labels
        self.use_coord = use_coord
        self.transforms = transforms
        self.normalize = normalize
        self.augmentation = augmentation
        self.debug = debug
        self.max_ninstance = max_ninstance
        # Calculate the valid range of lower left 3d corner to sample for a given output_size. if input patch_size
        # larger than original, fill with 0s
        self.MAX_SAMPLE_X = 140 - self.output_size[0]
        self.MAX_SAMPLE_Y = 140 - self.output_size[1]
        self.MAX_SAMPLE_Z = 1166 - self.output_size[2]

    def __iter__(self):
        """Sample batch_size number of worms, fetch data, do elastic augmentation, relabel them according to cpm,
        add coordination channels if use_coord=true, transform to torch.Tensor"""
        if self.train:
            while True:
                sampled_worms = random.sample(self.worms_data, k=self.n_consistent_worms)

                # sample patch corner from valid points
                corner_output = (random.randint(0, self.MAX_SAMPLE_X),
                          random.randint(0, self.MAX_SAMPLE_Y),
                          random.randint(0, self.MAX_SAMPLE_Z))
                # sampled_patch_output = (slice(corner_output[0], corner_output[0] + self.output_size[0]),
                #                         slice(corner_output[1], corner_output[1] + self.output_size[1]),
                #                         slice(corner_output[2], corner_output[2] + self.output_size[2]))

                patch_output_diff = np.array(self.patch_size) - np.array(self.output_size)
                corner_input = np.array((corner_output[0] - patch_output_diff[0]//2,
                          corner_output[1] - patch_output_diff[1]//2,
                          corner_output[2] - patch_output_diff[2]//2))
                other_corner_input = np.array(list(corner_input[i]+self.patch_size[i] for i in range(
                    len(self.patch_size))))

                valid_corner_input = np.array(list(corner_input[i] if corner_input[i]>0 else 0 for i in range(len(
                    corner_input))))
                outofbound_corner_input = np.array(list(abs(corner_input[i]) if corner_input[i] < 0 else 0 for i in
                                                            range(len(corner_input))))
                valid_other_corner_input = np.array([
                    other_corner_input[0] if other_corner_input[0] < 140 else 140,
                    other_corner_input[1] if other_corner_input[1] < 140 else 140,
                    other_corner_input[2] if other_corner_input[2] < 1166 else 1166,
                    ])
                sampled_patch_input = (slice(valid_corner_input[0], valid_other_corner_input[0]),
                                        slice(valid_corner_input[1], valid_other_corner_input[1]),
                                        slice(valid_corner_input[2], valid_other_corner_input[2]))
                filled_patch_input = (slice(outofbound_corner_input[0], outofbound_corner_input[
                    0]+valid_other_corner_input[0] - valid_corner_input[0]),
                                      slice(outofbound_corner_input[1], outofbound_corner_input[
                                          1] + valid_other_corner_input[1] - valid_corner_input[1]),
                                      slice(outofbound_corner_input[2], outofbound_corner_input[
                                          2] + valid_other_corner_input[2] - valid_corner_input[2]))

                center_ouput_corner = np.array((self.patch_size-self.output_size)//2)
                center_output_patch_from_input = (slice(center_ouput_corner[0], center_ouput_corner[0]+self.output_size[0]),
                                                  slice(center_ouput_corner[1], center_ouput_corner[
                                                      1]+self.output_size[1]),
                                                  slice(center_ouput_corner[2], center_ouput_corner[
                                                      2]+self.output_size[2]),)


                raws = []
                original_raws = []
                seghyps = []
                original_seghyps = []
                unique_patch_labels = []
                wuids = []
                worm_fn_list = []
                for worm_fn in sampled_worms:
                    worm_fn_list.append(worm_fn)
                    wuid = int(worm_fn.split('/')[-1].split('.')[0].split('worm')[1])
                    wuids.append(wuid)
                    with h5py.File(worm_fn, 'r') as f:
                        # for input (raw) we need a larger patch than output possibly with 0 padding
                        raw_ = f['volumes/raw'][()][sampled_patch_input].astype('float')
                        raw = np.zeros(self.patch_size)
                        raw[filled_patch_input] = raw_
                        # Also for seghyps do the same as raw, so that no issues for augmentation, in the end,
                        # just take the most center patch of size patch_size
                        seghyp_ = f['volumes/nuclei_seghyp'][()][sampled_patch_input].astype('int')
                        seghyp = np.zeros(self.patch_size)
                        seghyp[filled_patch_input] = seghyp_

                    if self.debug:
                        original_seghyps.append(np.copy(seghyp))
                        original_raws.append(np.copy(raw))
                    # Do  data augmentation on raw and seghyp together
                    # TODO: augmentations should be implemented outside dataset. preferably by using transforms. or
                    #  mybe sth like torchio
                    if self.augmentation:
                        raw, seghyp = self._augment(raw, seghyp)

                    # after augmentation we are good to only take the necessary output patch size
                    seghyp = seghyp[center_output_patch_from_input]

                    raws.append(raw)
                    seghyps.append(seghyp)
                    ul = np.unique(seghyp).tolist()
                    if 0 in ul:
                        ul.remove(0)
                    unique_patch_labels.append(np.array(ul))

                # get a relabeling list based on cpm dataset, showing label mappings for every label in unique labels
                max_l, relabel_map_list = self._get_relabel_map(unique_patch_labels, wuids)
                if max_l < 2:  # having 0 or 1 n_instances in a sample is not valid. 1 instance at least in the
                    # discriminative loss does not make sense
                    continue

                # now we can relabel seghyp labels based on relabel_map_list
                tmp_seghyps = [np.zeros_like(seghyp) for seghyp in seghyps]
                for i, (relabel_map, original_label, seghyp) in enumerate(zip(relabel_map_list, unique_patch_labels,
                                                                         seghyps)):
                    for relabel, original_l in zip(relabel_map, original_label):
                        tmp_seghyps[i][seghyp==original_l] = relabel
                seghyps = tmp_seghyps

                # Add channel dimension to raw data
                for i, raw in enumerate(raws):
                    raws[i] = np.expand_dims(raw, axis=0)
                # Add coord channels to raw data based on corner and patchsize
                if self.use_coord:
                    xyz_coords = np.meshgrid(np.arange(start=corner_input[0], stop=corner_input[0]+self.patch_size[0]),
                                             np.arange(start=corner_input[1], stop=corner_input[1]+self.patch_size[1]),
                                             np.arange(start=corner_input[2], stop=corner_input[2]+self.patch_size[2]))
                    xyz_coords = [np.expand_dims(c, axis=0) for c in xyz_coords]
                    xyz_coords = np.vstack(xyz_coords)
                    for i, raw in enumerate(raws):
                        raws[i] = np.vstack([raw, xyz_coords])

                # TODO: the lines below could be added as transforms
                if self.transforms:
                    raise NotImplementedError()
                # expand seghyps into instance dimension, so that each instance is in its unique instance_dim.
                # Since we don't have a background instance, start from index 0. For now use this, to train models
                # easier, since this is the input to Disc loss implementation, and better to do it in the dataset
                # object where workers are available. TODO: plan to change this, because tensors too large to be able
                #  to make use of larger patch sizes
                tmp_seghyps = [np.zeros((max_l,) + seghyps[0].shape) for _ in seghyps]
                for i, seghyp in enumerate(seghyps):
                    ul = np.unique(seghyp).tolist()
                    ul = [int(uull) for uull in ul]
                    if 0 in ul:
                        ul.remove(0)
                    unique_labels = np.array(ul)
                    for l in unique_labels:
                        tmp_seghyps[i][l-1, seghyp==l] = 1

                seghyps = tmp_seghyps

                # Disc loss especially for very large n_instance cases, throws GPU OOM Error. While there has been
                # some improvements for Disc loss, it still remains an issue for larger n_instance sizes. E.g. for
                # patch size of [64, 64, 64], it throws OOM error for n_instance 140. However, it does not for 120.
                # Still havn't done an extensive study of the exact n_instance number, but use max_ninstance
                # parameter to randomely select max_ninstance number if it has more.
                if self.max_ninstance and self.max_ninstance != 0 and max_l > self.max_ninstance:
                    # for this, randomely select max_ninstance of [0, max_l - 1] and only take those rows
                    logger.debug(f'Clipping input seghyp data from original:[{max_l}] n_instances to [{self.max_ninstance}]')
                    selected_rows = np.random.choice(max_l, self.max_ninstance, replace=False)
                    for i, seghyp in enumerate(seghyps):
                        seghyps[i] = seghyp[selected_rows, :]
                    max_l = self.max_ninstance

                # Create batch dimensions from the lists
                for i, (raw, seghyp) in enumerate(zip(raws, seghyps)):
                    raws[i] = np.expand_dims(raw, axis=0)
                    seghyps[i] = np.expand_dims(seghyp, axis=0)
                raws = np.vstack(raws)
                seghyps = np.vstack(seghyps)

                # Normalize raw data
                if self.normalize:
                    raws[:, 0] /= 255.0
                    if self.use_coord:
                        raws[:, 1] = (raws[:, 1] - 70.0)/70.0
                        raws[:, 2] = (raws[:, 2] - 70.0) / 70.0
                        raws[:, 3] = (raws[:, 3] - 583.0) / 583.0

                # cast them to tensor objects
                sample = {'raw': raws,
                          'label': seghyps,
                          'n_cluster': np.array(max_l)}

                if self.debug:
                    sample.update({
                        'original_raw': np.vstack([np.expand_dims(a, axis=0) for a in original_raws]),
                        'original_label': np.vstack([np.expand_dims(a, axis=0) for a in original_seghyps]),
                        'corner': corner_input,
                        'worm_fn': worm_fn_list
                        })
                sample.update({k: torch.from_numpy(v) for k, v in sample.items() if type(v) is np.ndarray})
                yield sample

        else:  # self.train == False
            logger.error('Use this dataset only for training. Use WormsDatasetOverSeghypCenters for test instead.')
            raise ValueError()

    def _get_relabel_map(self, unique_patch_labels, wuids):
        # since cpm dataset is constructed in a way that wuid1<wuid2. let's sort all input lists based on wuids,
        # and later reorder them based on their original labels.
        compact_list_to_sort = zip(wuids, unique_patch_labels, range(len(wuids)))
        compact_list_to_sort = sorted(compact_list_to_sort)
        wuids, unique_patch_labels, reorder_idxs = zip(*compact_list_to_sort)

        relabel_map_list = [[0] * len(l) for l in unique_patch_labels]  # by initializing to 0 values, any label
        # not relabeled will be deleted

        if len(wuids) == 1:
            relabel_map_list = [[i for i, _ in enumerate(unique_patch_labels[0], start=1)]]
            max_l = len(unique_patch_labels[0])
        elif len(wuids) == 2:
            # for the case of only pair of worms, just access the corresponding cpm data, if cpm labels both present
            # in the two patches (unique_patch_labels), assign them the same label, if not, depending on
            # use_leftout_labels, assign these labels the next label in line. if it needs to be removed, assign 0 to
            # them
            wuid1, wuid2 = wuids[0], wuids[1]
            cpm12 = self.cpm_data[f'{wuid1}-{wuid2}']
            curr_relabel_id = 1

            # iterate over one side, relabel itself and the otherside accordingly, do the same for the other one,
            # don't relabel if it has been relabeled already (if it is 0 or not)
            for i, l1 in enumerate(unique_patch_labels[0]):
                increase = False
                if self.use_leftout_labels:
                    relabel_map_list[0][i] = curr_relabel_id
                    increase = True
                if l1 in cpm12.keys():
                    l2 = cpm12[l1]
                    if l2 in unique_patch_labels[1]:
                        i2 = unique_patch_labels[1].tolist().index(l2)
                        relabel_map_list[1][i2] = curr_relabel_id
                        relabel_map_list[0][i] = curr_relabel_id
                        increase = True
                if increase:
                    curr_relabel_id += 1
            for i, l2 in enumerate(unique_patch_labels[1]):
                increase = False
                if relabel_map_list[1][i] != 0:
                    continue
                if self.use_leftout_labels:
                    relabel_map_list[1][i] = curr_relabel_id
                    increase = True
                # since we have already investigated occuring labels in both patches in the previous loop,
                # no need to do it here anymore
                if increase:
                    curr_relabel_id += 1
            max_l = curr_relabel_id - 1

        else:
            raise NotImplementedError

        # use reorder_idxs to correctly reorder relabel_map_list
        _, relabel_map_list = zip(*sorted(zip(reorder_idxs, relabel_map_list)))
        return max_l, relabel_map_list

    def _augment(self, raw, seghyp):
        if self.augmentation.ELASTIC.P:
            p = random.random()
            if p < self.augmentation.ELASTIC.P:
                elastic_params = self.augmentation.ELASTIC
                # rotation interval in radian
                rot_interval = np.array(elastic_params.ROTATION_INTERVAL) * np.math.pi/180.0
                transformation = create_elastic_transformation(
                    raw.shape,
                    control_point_spacing=elastic_params.CONTROL_POINT_SPACING,
                    jitter_sigma=elastic_params.JITTER_SIGMA,
                    rotation_interval=rot_interval,
                    subsample=elastic_params.SUBSAMPLE)
                raw = augment.apply_transformation(raw, transformation)
                seghyp = augment.apply_transformation(seghyp, transformation, interpolate=False)
        return raw, seghyp


# TODO: This looks super STUPID. but for now
class WormsDatasetOverSeghypCenters:
    def __init__(self,
                 worms_dataset_root,
                 patch_size,
                 output_size,
                 use_coord,
                 normalize):
        # super(WormsDatasetOverSeghypCenters, self).__init__()
        self.worms_data = glob.glob(os.path.join(worms_dataset_root, "*.hdf"))
        self.patch_size = patch_size
        self.output_size = output_size
        self.use_coord = use_coord
        self.normalize = normalize

    def __len__(self):
        return len(self.worms_data)

    def __getitem__(self, idx):
        return OneWormDatasetOverSeghypCenters(self.worms_data[idx], patch_size=self.patch_size,
                                               output_size=self.output_size,
                                               use_coord=self.use_coord, normalize=self.normalize)


class OneWormDatasetOverSeghypCenters(Dataset):
    def __init__(self,
                 worm_data,
                 patch_size,
                 output_size,
                 use_coord,
                 normalize):
        super(OneWormDatasetOverSeghypCenters, self).__init__()
        self.worm_data = worm_data
        self.patch_size = np.array(patch_size)
        self.output_size = np.array(output_size)
        self.use_coord = use_coord
        self.normalize = normalize
        with h5py.File(self.worm_data, 'r') as f:
            self.raw = f['volumes/raw'][()].astype('float')
            self.seghyp = f['volumes/nuclei_seghyp'][()].astype('int')
            self.con_seghyp = f['matrix/con_seghyp'][()]
            self.gt_label = f['volumes/gt_nuclei_labels'][()].astype('int')
        self.full_size = self.raw.shape

    def __len__(self):
        return self.con_seghyp.shape[0]-1  # first one is background label 0 with con [0, 0, 0]

    def __getitem__(self, tmp_idx):
        idx = tmp_idx + 1
        con_idx = np.round(self.con_seghyp[idx]).astype(dtype=np.int)
        if any(con_idx<-10000) or any(con_idx>10000):
            # TODO: sometimes for worm18 con_idx has very large negative number, overflow
            logger.info(f'Skipped loading con-seghyp with index:[{idx}], worm_name:[{self.worm_data}]')
            return {'mask': torch.from_numpy(np.array([-1]))}

        # other_corner_extra = con_idx + self.patch_size//2 - self.full_size
        # other_corner_extra[other_corner_extra<0] = 0
        # corner = con_idx - self.patch_size//2 - other_corner_extra
        # corner[corner < 0] = 0
        # patch = (slice(corner[0], corner[0]+self.patch_size[0]),
        #          slice(corner[1], corner[1]+self.patch_size[1]),
        #          slice(corner[2], corner[2]+self.patch_size[2]))

        raw, corner = extract_0padded_patch(self.patch_size, con_idx, self.raw)
        # raw = self.raw[patch]
        raw = np.expand_dims(raw, axis=0)  # add channel dim
        if self.use_coord:
            xyz_coords = np.meshgrid(np.arange(start=corner[0], stop=corner[0] + self.patch_size[0]),
                                     np.arange(start=corner[1], stop=corner[1] + self.patch_size[1]),
                                     np.arange(start=corner[2], stop=corner[2] + self.patch_size[2]))
            xyz_coords = [np.expand_dims(c, axis=0) for c in xyz_coords]
            xyz_coords = np.vstack(xyz_coords)
            raw = np.vstack([raw, xyz_coords])
        if self.normalize:
            raw[0] /= 255.0
            if self.use_coord:
                raw[1] = (raw[1] - 70.0) / 70.0
                raw[2] = (raw[2] - 70.0) / 70.0
                raw[3] = (raw[3] - 583.0) / 583.0

        masktmp, _ = extract_0padded_patch(self.output_size, con_idx, self.seghyp)
        # masktmp = self.seghyp[patch]
        mask = np.zeros_like(masktmp)
        mask[masktmp==idx] = 1
        gt_label,_ = extract_0padded_patch(self.output_size, con_idx, self.gt_label)
        # gt_label = self.gt_label[patch]
        gt_label_ids = gt_label[mask==1]
        if len(gt_label_ids)==0:
            # TODO: clean the dataset for worm18 to not have this
            logger.info(f'Skipped loading con-seghyp with index:[{idx}], worm_name:[{self.worm_data}]')
            return {'mask': torch.from_numpy(np.array([-1]))}
        gt_label_id = int(gt_label_ids[0])
        assert np.all(gt_label_ids == gt_label_id)
        gt_label_id = np.array(gt_label_id)
        sample = {'patch_reference_corner': corner,
                  'raw': raw,
                  'mask': mask,
                  'gt_label_id': gt_label_id}
        sample = {k: torch.from_numpy(v) for k, v in sample.items()}
        return sample


def extract_0padded_patch(shape, con, arr):
    con = np.array(con)
    shape = np.array(shape)
    assert len(con) == 3
    corner = con - shape//2
    valid_corner = np.array([corner[i] if corner[i]>=0 else 0 for i in range(len(corner))])
    other_corner = corner + shape
    valid_other_corner = np.array(list(other_corner[i] if other_corner[i]<=ll else ll for i, ll in enumerate([140,
                                                                                                              140,
                                                                                                              1166])))

    relative_patch = (
        slice(valid_corner[0]-corner[0], shape[0] - other_corner[0]+valid_other_corner[0]),
        slice(valid_corner[1] - corner[1], shape[1] - other_corner[1] + valid_other_corner[1]),
        slice(valid_corner[2] - corner[2], shape[2] - other_corner[2] + valid_other_corner[2]),
    )

    sampled_patch = (
        slice(valid_corner[0], valid_other_corner[0]),
        slice(valid_corner[1], valid_other_corner[1]),
        slice(valid_corner[2], valid_other_corner[2]),
    )

    if sampled_patch[0].start > 1000:
        print('jj')
    raw = np.zeros(shape)
    raw[relative_patch] = arr[sampled_patch]

    return raw, corner