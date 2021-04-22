import logging
import time
import os

import torch
import torch.nn as nn
import torchsummary
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from joblib import dump, load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import plotly.offline as plotlyoff
import plotly.graph_objs as plotlygo

from funlib.learn.torch.models import UNet

from lib.models.pixelwise_model import PixelwiseModel
from lib.modules.discriminative_loss_valid_push_force import DiscriminativeLossValidPushForce


logger = logging.getLogger(__name__)


class UnetValidPushForceDiscLoss(PixelwiseModel):
    def __init__(self,
                 sett,
                 load_model_path=None):
        super(UnetValidPushForceDiscLoss, self).__init__(sett, load_model_path)

    def _define_model(self, sett):
        return UNet(
            in_channels= 1 if not sett.DATA.USE_COORD else 4,
            num_fmaps=sett.MODEL.MODEL_PARAMS.NUM_FMAPS,
            fmap_inc_factor=sett.MODEL.MODEL_PARAMS.FMAP_INC_FACTOR,
            downsample_factors=sett.MODEL.MODEL_PARAMS.DOWNSAMPLE_FACTORS
            )

    def print_model_summary(self, sample_input, tb_writer=None):
        # TODO: summary only prints to stdout, does not return str for logger
        input_size = sample_input['raw'].size()[1:]
        input_size = (1,) + input_size
        print('Backbone Summary:')
        torchsummary.summary(self.model, input_size)

        if tb_writer:
            tb_writer.add_graph(self.model, torch.split(sample_input['raw'],1)[0].unsqueeze(0).float().cuda())
        # Do not need this, since torchsummary does not count weight sharing -> wrong trainable param information, etc.
        # print('Siamese  Summary:')
        # torchsummary.summary(self.model, [input_size, input_size])

    def _define_criterion(self):
        # TODO: currently only loss with default values
        return DiscriminativeLossValidPushForce()

    def _train_one_iteration(self, input, criterion, optimizer):
        raw = input['raw'].float().cuda()
        # funlib unet model data input is [batch, Channel, x,y,z] but my dataset gives [batch, x,y,x] because grey
        # values, here fix this
        raw = raw.unsqueeze(1)
        seghyp = input['seghyp'].float().cuda()
        valid_discloss_pushforce_matrix = input['valid_discloss_pushforce'].cuda()

        optimizer.zero_grad()
        out = self.model(raw)

        loss, loss_dict = criterion(out, seghyp, valid_discloss_pushforce_matrix)
        assert not torch.isnan(loss), 'Why is loss NaN sometimes?'
        if loss.item() > 100:
            logger.debug('SKIPPING TRAINING - NOT TRAINING FOR THIS STEP BECAUSE LOSS TOO LARGE')
            return loss, loss_dict

        loss.backward()
        optimizer.step()

        return loss, loss_dict