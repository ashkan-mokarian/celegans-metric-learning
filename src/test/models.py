import torch
from torchsummary import summary

import _init_paths

from lib.models.models import ConvNet

if __name__ == '__main__':
    net = ConvNet(
        [4, 16, 32, 64, 128],
        [2, 2, 2, 2],
        [1024, 256, 128]  # assuming 64x64x64 patches
        )
    input_patch = torch.randn(1, 4, 64, 64, 64)
    net = net.to("cuda:0")
    out = net(input_patch.cuda())
    outn = out.cpu().detach().numpy()
    print(summary(net, input_patch.size()[1:]))
    print("finished")