from __future__ import print_function
import pprint
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type, ones_like
from models.layers import View, Identity
from models.gumbel import GumbelSoftmax
from models.isotropic_gaussian import IsotropicGaussian


class Mixture(nn.Module):
    ''' gaussian + discrete reparaterization '''
    def __init__(self, num_discrete, num_continuous, config):
        super(Mixture, self).__init__()
        self.config = config
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous*2

        # setup the gaussian config
        gaussian_config = copy.deepcopy(config)
        gaussian_config['latent_size'] = num_continuous
        self.gaussian = IsotropicGaussian(gaussian_config)

        # setup the discrete config
        discrete_config = copy.deepcopy(config)
        discrete_config['latent_size'] = num_discrete
        self.discrete = GumbelSoftmax(discrete_config)

        self.output_size = self.discrete.output_size + self.gaussian.output_size

    def prior(self, shape):
        batch_size = shape[0]
        disc = self.discrete.prior([batch_size, self.num_discrete])
        cont = self.gaussian.prior([batch_size, self.num_continuous])
        return torch.cat([cont, disc], 1)

    def mutual_info(self, params, eps=1e-9):
        q_z_given_x = params['discrete']['q_z']
        batch_size = q_z_given_x.size(0)
        n_features = q_z_given_x.size(-1)
        cat = torch.distributions.Categorical(
            ones_like(q_z_given_x, self.config['cuda']).div_(n_features)
        )
        p_z = cat.sample_n(batch_size) # prior sample

        crossent_loss = torch.mean(-torch.sum(p_z * torch.log(q_z_given_x + eps), dim=1))
        ent_loss = torch.mean(-torch.sum(p_z * torch.log(p_z + eps), dim=1))
        return crossent_loss + ent_loss

    def reparmeterize(self, logits):
        gaussian_logits = logits[:, 0:self.num_continuous]
        discrete_logits = logits[:, self.num_continuous:]

        gaussian_reparam, gauss_params = self.gaussian(gaussian_logits)
        discrete_reparam, disc_params = self.discrete(discrete_logits)

        merged = torch.cat([gaussian_reparam, discrete_reparam], -1)
        params = {'gaussian': gauss_params,
                  'discrete': disc_params,
                  'z': merged}
        return merged, params

    def kl(self, dist_a):
        gauss_kl = self.gaussian.kl(dist_a['gaussian'])
        disc_kl = self.discrete.kl(dist_a['discrete'])
        return gauss_kl + disc_kl

    def forward(self, logits):
        return self.reparmeterize(logits)
