import torch
import torch.utils.data
from torchsummary import summary

import _init_paths

from lib.modules.unet import UNet
from lib.models.siamese_pixelwise_model import SiamesePixelwiseModel
from lib.data.siamese_worms_dataset import SiameseWormsDataset

from src.scripts.settings import Settings

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    unet = UNet(4, 64, 4, 3)

    ts = Settings('siameseunet_test', 'train_default')

    train_dataset = SiameseWormsDataset(ts.PATH.WORMS_DATASET,
                                        ts.PATH.CPM_DATASET,
                                        patch_size=(140, 140, 140),
                                        train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=10
        )

    print('Checking Siamese memory trace')
    torch.cuda.empty_cache()
    siamese_unet = SiamesePixelwiseModel(unet).cuda()
    input_patch = torch.randn(1,4, 104, 104, 104)
    input2_patch = input_patch.clone()
    out1, out2 = siamese_unet.forward(input_patch.cuda(), input2_patch.cuda())
    input_size = input_patch.size()[1:]
    print(summary(siamese_unet, [input_size, input_size]))
    print("finished")