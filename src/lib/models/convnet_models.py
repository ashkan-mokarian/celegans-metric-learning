import torch.nn as nn


class ConvNet(nn.Module):
    """Typical simple stack of convolutional layers followed by FC layers"""
    def __init__(
            self,
            fmaps,
            downsample_factors,
            fc_dims,
            activation="ReLU",
            final_activation=False,
            **kwargs):
        super(ConvNet, self).__init__()

        assert len(fmaps)-1 == len(downsample_factors), \
            "len of fmaps does not match with len of downsample_factors." \
            "fmaps[0] is the number of in_channels"
        activation = getattr(nn, activation)
        convnet = nn.ModuleList()
        for l in range(1, len(fmaps)):
            convlayer = nn.Sequential(
                nn.Conv3d(fmaps[l-1], fmaps[l], 3),
                activation(),
                nn.MaxPool3d(downsample_factors[l-1], stride=downsample_factors[l-1])
                )
            convnet.append(convlayer)
        self.convnet = nn.Sequential(*convnet)

        fc = nn.ModuleList()
        for l in range(1, len(fc_dims)):
            fc.append(nn.Linear(fc_dims[l-1], fc_dims[l]))
            if l < len(fc_dims)-1 or final_activation:
                fc.append(activation())
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


class ConvNetL2(ConvNet):
    """same as ConvNet with L2-normalized embeddings"""
    def __init__(self):
        super(ConvNetL2, self).__init__()

    def forward(self, x):
        output = super(ConvNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


# class ClassificationNet(nn.Module):
#     """Wrapper for an embedding network, adds a FC layer with Log Softmax
#     for classification"""
#     def __init__(self, embedding_net, n_classes):
#         super(ClassificationNet, self).__init__()
#         self.embedding_net = embedding_net
#         self.n_classes = n_classes
#         self.nonlinear = nn.ReLU()
#         self.fc1 = nn.Linear(2, n_classes)
#
#     def forward(self, x):
#         output = self.embedding_net(x)
#         output = self.nonlinear(output)
#         scores = F.log_softmax(self.fc1(output), dim=-1)
#         return scores
#
#     def get_embedding(self, x):
#         return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    """Wrapper for an embedding network, Processes 2 inputs"""
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    """Wrapper for an embedding network, processes 3 inputs"""
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)