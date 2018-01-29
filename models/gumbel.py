from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type, one_hot
from models.layers import View, Identity, UpsampleConvLayer


class GumbelSoftmax(nn.Module):
    def __init__(self, config):
        super(GumbelSoftmax, self).__init__()
        self._setup_anneal_params()
        self.iteration = 0
        self.config = config
        self.input_size = self.config['discrete_size']
        self.output_size = self.config['discrete_size']

    def prior(self, shape):
        uniform_probs = float_type(self.config['cuda'])(1, shape[1]).zero_()
        uniform_probs += 1.0 / shape[1]
        cat = torch.distributions.Categorical(uniform_probs)
        sample = cat.sample_n(shape[0])
        return Variable(
            one_hot(shape[1], sample, use_cuda=self.config['cuda'])
        ).type(float_type(self.config['cuda']))

    def _setup_anneal_params(self):
        # setup the base gumbel rates
        # TODO: parameterize this
        self.tau = 1.0
        self.tau0 = 1.0
        self.anneal_rate = 0.0003
        # self.anneal_rate = 0.0003 #1e-5
        self.min_temp = 0.5

    def anneal(self):
        ''' Helper to anneal the categorical distribution'''
        if self.training is True \
           and self.iteration > 0 \
           and self.iteration % 1 == 0: # was originally 10

            # smoother annealing
            rate = -self.anneal_rate * self.iteration
            self.tau = np.maximum(self.tau0 * np.exp(rate),
                                  self.min_temp)
            # print("tau = ", self.tau)

            # hard annealing
            # self.tau = np.maximum(0.9 * self.tau, self.min_temp)

    def reparmeterize(self, logits):
        log_q_z = F.log_softmax(logits, dim=-1)
        z, z_hard = self.sample_gumbel(logits, self.tau,
                                       hard=True,
                                       use_cuda=self.config['cuda'])
        return z, z_hard, log_q_z

    @staticmethod
    def _kld_categorical_uniform(log_q_z, eps=1e-9):
        latent_size = log_q_z.size(-1)
        p_z = 1.0 / latent_size
        log_p_z = np.log(p_z + eps)
        kld_element = log_q_z.exp() * (log_q_z - log_p_z)
        return torch.sum(kld_element, dim=-1)

    def kl(self, dist_a):
        return GumbelSoftmax._kld_categorical_uniform(
            dist_a['discrete']['log_q_z'],
        )

    @staticmethod
    def _gumbel_softmax(x, tau, eps=1e-9, use_cuda=False):
        noise = torch.rand(x.size())
        # -ln(-ln(U + eps) + eps)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if use_cuda:
            noise = noise.cuda()

        noise = Variable(noise)
        x = (x + noise) / tau
        x = F.softmax(x.view(x.size(0), -1) + eps, dim=-1)
        return x.view_as(x)

    @staticmethod
    def sample_gumbel(x, tau, hard=False, use_cuda=True):
        y = GumbelSoftmax._gumbel_softmax(x, tau, use_cuda=use_cuda)

        if hard:
            y_max, _ = torch.max(y, dim=y.dim() - 1,
                                 keepdim=True)
            y_hard = Variable(
                torch.eq(y_max.data, y.data).type(float_type(use_cuda))
            )
            y_hard_diff = y_hard - y
            y_hard = y_hard_diff.detach() + y
            return y.view_as(x), y_hard.view_as(x)

        return y.view_as(x), None

    def forward(self, logits):
        self.anneal()  # anneal first
        z, z_hard, log_q_z = self.reparmeterize(logits)
        params = {
            'z_hard': z_hard,
            'logits': logits,
            'log_q_z': log_q_z
        }
        self.iteration += 1

        # return the reparameterization
        # and the params of gumbel
        return z, { 'z': z, 'discrete': params }
