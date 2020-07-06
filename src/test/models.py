import torch
from torchsummary import summary

import _init_paths

from lib.modules.unet import UNet
from lib.modules.convnet_models import SiameseNet

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    net = UNet(
        4,
        64,
        4,
        3
        )
    input_patch = torch.randn(1, 4, 100, 100, 100)
    net = net.to("cuda:0")
    out = net(input_patch.cuda())
    print(summary(net, input_patch.size()[1:]))

    print('Checking Siamese memory trace')
    torch.cuda.empty_cache()
    siamese_unet = SiameseNet(net)
    input2_patch = torch.randn(1,4, 100, 100, 100)
    out1, out2 = siamese_unet(input_patch.cuda(), input2_patch.cuda())
    input_size = input_patch.size()[1:]
    print(summary(siamese_unet, [input_size, input_size]))
    print("finished")