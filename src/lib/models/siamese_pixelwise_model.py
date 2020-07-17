import logging
import time

import os

import torch
import torch.nn as nn
import torchsummary
import numpy as np

from lib.modules.unet import UNet
from lib.modules.discriminative_loss import DiscriminativeLoss

logger = logging.getLogger(__name__)


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


class SiamesePixelwiseModel:
    """Defines training, evaluation, ... for a **siamese pixel-wise model**, e.g. siamese_unet"""
    def __init__(self,
                 backbone_model_name, backbone_model_params,
                 load_model_path=None):

        if backbone_model_name == 'unet':
            unet = UNet(*backbone_model_params)
        else:
            raise NotImplementedError()
        self.model = SiameseNet(unet)
        self.best_loss = np.inf
        self.step = 0
        self.load_model_path = load_model_path
        if load_model_path:
            assert os.path.exists(load_model_path), f'load_model_path:[{load_model_path}], does not exist.'
            self.step, self.best_loss = self.load_model(load_model_path)
        self.model.cuda()

    def save_model(self, step, optimizer, lr_scheduler, loss, filename):
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss
            }, filename)
        logger.info(f'Saved model to:[{filename}]. step:[{step}]. loss:[{loss}]')

    def load_model(self, filename, optimizer=None, lr_scheduler=None):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        step = checkpoint['step']
        loss = checkpoint['loss']
        logger.info(f'Loaded model from:[{filename}]. step:[{step}]. loss:[{loss}]')
        return step, loss

    def print_model_summary(self, sample_data):
        # TODO: summary only prints to stdout, does not return str for logger
        input_size = sample_data['raw1'].size()[1:]
        print('Backbone Summary:')
        torchsummary.summary(self.model.embedding_net, input_size)
        # Do not need this, since torchsummary does not count weight sharing -> wrong trainable param information, etc.
        # print('Siamese  Summary:')
        # torchsummary.summary(self.model, [input_size, input_size])

    @staticmethod
    def _define_criterion():
        # TODO: currently only loss with default values
        return DiscriminativeLoss()

    def _define_optimizer(self, learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, load_state=True):
        parameters = self.model.parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=lr_drop_factor, patience=lr_drop_patience, verbose=True
            )
        if load_state:
            if self.load_model_path:
                checkpoint = torch.load(self.load_model_path)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        return optimizer, lr_scheduler

    def _train_one_iteration(self, inputs, criterion, optimizer):
        raw1 = inputs['raw1'].float().cuda()
        raw2 = inputs['raw2'].float().cuda()
        label1 = inputs['label1'].cuda()
        label2 = inputs['label2'].cuda()
        n_cluster = inputs['n_cluster'].cuda()

        optimizer.zero_grad()
        out1, out2 = self.model(raw1, raw2)

        # How to do discriminative loss, for a siamese network?
        # Easy way would be to just concatenate the two patches, in the input, before forwarding it to the model.
        # and label each pixel accordingly, and this way, we have the two pixels from the two patches labeled as
        # same class and therefore everything is fine. However,
        # by concatenating the input, there is the boundary effect, especially where the two patches touch,
        # therefore the embedding for the pixels would be affected. I didn't know any other workaround,
        # other than running them seperately through the network, and after the two pixel-wise feature vectors
        # are computed seperately, now concatenate the two patches, since there are no FOV s anymore in the
        # computations.

        # Therefore, we first change the view (losing spatial information, but not used in loss anyways)
        # out1, out2 must have same sizes. label1, label2 must have same sizes. therefore getting size of one
        # should be enough
        bs, n_features, xsize, ysize, zsize = out1.size()
        out1 = out1.contiguous().view(bs, n_features, xsize * ysize * zsize)
        out2 = out2.contiguous().view(bs, n_features, xsize * ysize * zsize)
        out = torch.cat((out1, out2), dim=-1)
        max_n_clusters = label1.size(1)
        label1 = label1.contiguous().view(bs, max_n_clusters, xsize * ysize * zsize)
        label2 = label2.contiguous().view(bs, max_n_clusters, xsize * ysize * zsize)
        label = torch.cat((label1, label2), dim=-1)

        loss = criterion(out, label, n_cluster)
        assert not torch.isnan(loss), 'Why is loss NaN sometimes?'

        loss.backward()
        optimizer.step()

        return loss

    def fit(self, train_loader, n_step, learning_rate, weight_decay, lr_drop_factor, lr_drop_patience,
            model_ckpt_every_n_step, model_save_path, runnin_loss_interval, burn_in_step, tb_writer=None):
        criterion = self._define_criterion()
        optimizer, lr_scheduler = self._define_optimizer(learning_rate, weight_decay, lr_drop_factor, lr_drop_patience)

        logger.info(f'Start training from step:{self.step} to step:{n_step}')

        train_iter = iter(train_loader)

        if tb_writer:
            tmp_input = next(train_iter)
            tb_writer.add_graph(self.model, [tmp_input['raw1'].float().cuda(), tmp_input['raw2'].float().cuda()])

        self.model.train()

        running_loss_timer = time.time()
        running_loss = np.zeros(runnin_loss_interval)
        for step in range(self.step+1, n_step+1):
            inputs = next(train_iter)
            loss = self._train_one_iteration(inputs, criterion, optimizer)

            running_loss[:-1] = running_loss[1:]; running_loss[-1] = loss.item()

            tb_writer.add_scalar('Loss/train', loss.item(), step)
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

            if step % model_ckpt_every_n_step == 0:
                self.save_model(step, optimizer, lr_scheduler, running_loss.mean(),
                                os.path.join(model_save_path, f'model-step={step}.pth'))

            if step % runnin_loss_interval == 0:
                avg_running_loss = running_loss.mean()
                if avg_running_loss < self.best_loss:
                    self.best_loss = avg_running_loss
                    if step > burn_in_step:
                        # Saving best model
                        existing_best_model = [os.path.join(model_save_path, f) for f in os.listdir(model_save_path) if
                                               f.startswith('bestmodel')]
                        if existing_best_model:
                            os.remove(existing_best_model[0])
                        self.save_model(step, optimizer, lr_scheduler, avg_running_loss,
                                        os.path.join(model_save_path, f'bestmodel-step={step}-running_loss='
                                                                      f'{avg_running_loss:.4f}.pth'))

                # preferably do it every epoch, but since no epoch meaning, use running_loss_interval instead
                if tb_writer:
                    for param_name, param in self.model.named_parameters():
                        tb_writer.add_histogram(param_name, param.data, step)

                # TODO: best case scenario should be a validation loss or metric of some sort, but running_loss for now
                lr_scheduler.step(avg_running_loss)

                logger.info(f'steps [{step}/{n_step}] - running_loss [{avg_running_loss:.5f}] - time ['
                            f'{time.time()-running_loss_timer:.2f}s]')
                running_loss_timer = time.time()


