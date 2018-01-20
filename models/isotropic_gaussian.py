from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type
from models.layers import View, Identity


class IsotropicGaussian(nn.Module):
    ''' isotropic gaussian reparameterization '''
    def __init__(self, config):
        super(IsotropicGaussian, self).__init__()
        self.config = config
        self.input_size = self.config['continuous_size']
        assert self.config['continuous_size'] % 2 == 0
        self.output_size = self.config['continuous_size'] // 2

    def prior(self, shape):
        return Variable(
            float_type(self.config['cuda'])(*shape).normal_()
        )

    def _reparametrize_gaussian(self, mu, logvar):
#        if self.training:
            std = logvar.mul(0.5).exp_()
            #std = logvar.sqrt()
            eps = float_type(self.config['cuda'])(std.size()).normal_()
            eps = Variable(eps)
            return eps.mul(std).add_(mu), {'mu': mu, 'logvar': logvar}

 #       return mu, {'mu': mu, 'logvar': logvar}

    def reparmeterize(self, logits):
        feature_size = logits.size(-1)
        assert feature_size % 2 == 0
        mu = logits[:, 0:int(feature_size/2)]
        #sigma = F.softplus(logits[:, int(feature_size/2):])
        sigma = logits[:, int(feature_size/2):]
        return self._reparametrize_gaussian(mu, sigma)

    @staticmethod
    def _kld_gaussian_N_0_1(mu, logvar, feat_size, batch_size):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_element /= batch_size * feat_size
        return kld_element

        # kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # #return torch.mean(kld_element).mul_(-0.5)
        # return torch.sum(kld_element).mul_(-0.5)

    def kl(self, dist_a):
        return IsotropicGaussian._kld_gaussian_N_0_1(
            dist_a['mu'], dist_a['logvar'],
            batch_size=dist_a['mu'].size(0),
            feat_size=int(np.prod(self.config['img_shp']))
        )

    def forward(self, logits):
        return self.reparmeterize(logits)
