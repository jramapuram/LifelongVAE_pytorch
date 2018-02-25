from __future__ import print_function
import pyro
import torch.nn as nn
import pyro.distributions as D
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type, zeros_like, ones_like


class IsotropicGaussian(nn.Module):
    ''' isotropic gaussian reparameterization '''
    def __init__(self, config):
        super(IsotropicGaussian, self).__init__()
        self.config = config
        self.input_size = self.config['continuous_size']
        assert self.config['continuous_size'] % 2 == 0
        self.output_size = self.config['continuous_size'] // 2

    def prior(self, shape):
        mu = Variable(
            float_type(self.config['cuda'])(*shape).zeros_()
        )
        sigma = Variable(
            float_type(self.config['cuda'])(*shape).zeros_() + 1
        )
        return pyro.sample("continuous", D.Normal(mu, sigma))

    def _reparametrize_gaussian(self, mu, logvar):
        z =  pyro.sample("continuous", D.Normal(mu, logvar))
        if self.training:
            return z, {'mu': mu, 'logvar': logvar}

        # at test time we want only the mean
        return pyro.sample("continuous", lambda mu=mu: mu), \
            {'mu': mu, 'logvar': logvar}

    def reparmeterize(self, logits):
        feature_size = logits.size(-1)
        assert feature_size % 2 == 0
        mu = logits[:, 0:int(feature_size/2)]
        sigma = F.softplus(logits[:, int(feature_size/2):]) + 1e-6
        return self._reparametrize_gaussian(mu, sigma)

    def forward(self, logits):
        z, gauss_params = self.reparmeterize(logits)
        return z, { 'z': z, 'gaussian':  gauss_params }
