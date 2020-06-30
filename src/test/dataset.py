from lib.data.one_pos_rest_neg_patch_dataset import OnePosRestNegPatchDataset


if __name__ == '__main__':
    print('testing...')
    data_root = "/home/ashkan/workspace/deployed/worms_nuclei_metric_learning-deployed/data/raw/30WormsImagesGroundTruthSeg"
    pos_data = '/home/ashkan/workspace/deployed/worms_nuclei_metric_learning-deployed/data/processed/pairwise_matching_dataset.pkl'
    train_data = OnePosRestNegPatchDataset(data_root, pos_data, 40)
    print("Finished!!!")