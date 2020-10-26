"""
This is the implementation of following paper:
https://arxiv.org/pdf/1802.05591.pdf
This implementation is based on following code:
https://github.com/Wizaron/instance-segmentation-pytorch
"""
import logging

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch

logger = logging.getLogger(__name__)


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0,
                 beta=1.0, gamma=0.001, usegpu=True):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):
        target.detach_()
        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, input, target):
        bs, n_features, xsize, ysize, zsize = input.size()
        n_clusters = target.size(1)
        # Since no typical batch meaning here, consider it as extra pixel wise embeddings. yes, spatial locality
        # lost, but we don't need it from here on
        n_loc = bs*xsize*ysize*zsize
        input = input.transpose_(0, 1).contiguous().view(n_features, n_loc)
        target = target.transpose_(0, 1).contiguous().view(n_clusters, n_loc)

        c_means = self._cluster_means(input, target)
        l_var = self._variance_term(input, target, c_means)
        l_dist = self._distance_term(c_means)
        l_reg = self._regularization_term(c_means)
        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

        logger.debug(f'l_var:[{l_var}], l_dist:[{l_dist}], l_reg:[{l_reg}], loss:[{loss}]')

        return loss, l_var, l_dist, l_reg

    def _cluster_means(self, input, target):
        n_features, n_loc = input.size()
        n_clusters = target.size(0)
        # n_features, n_clusters, n_loc
        #input = input.unsqueeze(1).expand(n_features, n_clusters, n_loc)
        # 1, n_clusters, n_loc
        #target = target.unsqueeze(0)
        # n_features, n_clusters, n_loc
        #input = input * target

        # n_features, n_cluster
        #mean = input.sum(2) / target.sum(2)

        count = target.sum(1)
        sum = input @ target.transpose(0,1)
        mean = torch.true_divide(sum, count)

        return mean

    def _variance_term(self, input, target, c_means):
        n_features, n_loc = input.size()
        n_clusters = target.size(0)
        # n_features, n_clusters, n_loc
        c_means = c_means.unsqueeze(2).expand(n_features, n_clusters, n_loc)
        # n_features, n_clusters, n_loc
        input = input.unsqueeze(1).expand(n_features, n_clusters, n_loc)
        # n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 0) -
                           self.delta_var, min=0) ** 2) * target

        # n_clusters
        # print('var_sum',var.sum(1),'target_sum', target.sum(1))
        c_var = var.sum(1) / target.sum(1)
        var_term = c_var.sum() / n_clusters

        return var_term

    def _distance_term(self, c_means):
        n_features, n_clusters = c_means.size()

        # n_features, n_clusters, n_clusters
        means_a = c_means.unsqueeze(2).expand(n_features, n_clusters, n_clusters)
        means_b = means_a.permute(0, 2, 1)
        diff = means_a - means_b

        margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters))
        margin = Variable(margin)
        if self.usegpu:
            margin = margin.cuda()
        c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
        dist_term = c_dist / (2 * n_clusters * (n_clusters - 1))

        return dist_term

    def _regularization_term(self, c_means):
        n_features, n_clusters = c_means.size()
        reg_term = torch.mean(torch.norm(c_means, self.norm, 0))
        return reg_term
