import torch
import torch.utils.data
from torchsummary import summary

import _init_paths

from lib.modules.unet import UNet
from lib.models.pixelwise_model import PixelwiseModel
from lib.data.worms_dataset import WormsDataset

from src.scripts.settings import Settings

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    unet = UNet(4, 64, 4, 3)

    ts = Settings('siameseunet_test', 'train_default')

    train_dataset = WormsDataset(
        ts.PATH.WORMS_DATASET,
        ts.PATH.CPM_DATASET,
        patch_size=(64, 64, 64),
        n_consistent_worms=2,
        use_leftout_labels=True,
        use_coord=True,
        normalize=True,
        augmentation=None,
        transforms=None,
        train=True,
        debug=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1
        )

    print('Checking Siamese memory trace')
    torch.cuda.empty_cache()
    siamese_unet = PixelwiseModel(unet).cuda()
    input_patch = torch.randn(1,4, 104, 104, 104)
    input2_patch = input_patch.clone()
    out1, out2 = siamese_unet.forward(input_patch.cuda(), input2_patch.cuda())
    input_size = input_patch.size()[1:]
    print(summary(siamese_unet, [input_size, input_size]))
    print("finished")