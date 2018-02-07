from __future__ import print_function
import pprint
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type, ones_like
from models.gumbel import GumbelSoftmax
from models.isotropic_gaussian import IsotropicGaussian


class Mixture(nn.Module):
    ''' gaussian + discrete reparaterization '''
    def __init__(self, num_discrete, num_continuous, config):
        super(Mixture, self).__init__()
        self.config = config
        self.num_discrete_input = num_discrete
        self.num_continuous_input = num_continuous

        # setup the gaussian config
        # gaussian_config = copy.deepcopy(config)
        # gaussian_config['latent_size'] = num_continuous
        self.gaussian = IsotropicGaussian(config)

        # setup the discrete config
        # discrete_config = copy.deepcopy(config)
        # discrete_config['latent_size'] = num_discrete
        self.discrete = GumbelSoftmax(config)

        self.input_size = num_continuous + num_discrete
        self.output_size = self.discrete.output_size + self.gaussian.output_size

    def prior(self, shape):
        batch_size = shape[0]
        disc = self.discrete.prior([batch_size, self.discrete.output_size])
        cont = self.gaussian.prior([batch_size, self.gaussian.output_size])
        return torch.cat([cont, disc], 1)

    def mutual_info(self, params):
        return self.discrete.mutual_info(params)

    def log_likelihood(self, z, params):
        cont = self.gaussian.log_likelihood(z[:, 0:self.gaussian.output_size], params)
        disc = self.discrete.log_likelihood(z[:, self.gaussian.output_size:], params)
        return torch.cat([cont, disc], 1)

    def reparmeterize(self, logits):
        gaussian_logits = logits[:, 0:self.num_continuous_input]
        discrete_logits = logits[:, self.num_continuous_input:]

        gaussian_reparam, gauss_params = self.gaussian(gaussian_logits)
        discrete_reparam, disc_params = self.discrete(discrete_logits)
        merged = torch.cat([gaussian_reparam, discrete_reparam], -1)

        params = {'gaussian': gauss_params['gaussian'],
                  'discrete': disc_params['discrete'],
                  'z': merged}
        return merged, params

    def kl(self, dist_a):
        gauss_kl = self.gaussian.kl(dist_a)
        disc_kl = self.discrete.kl(dist_a)
        return torch.cat([gauss_kl, disc_kl], dim=1)

    def forward(self, logits):
        return self.reparmeterize(logits)
