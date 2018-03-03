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


class ConcatReparameterizer(nn.Module):
    ''' takes multiple reparameterizers and concats them to return
        a list of reparam(f(x)) for each reparam in the list'''
    def __init__(self, reparameterizer_list, config):
        super(ConcatReparameterizer, self).__init__()
        assert isinstance(reparameterizer_list, list)
        self.config = config
        self.reparmeterizers = reparameterizer_list

        # tabulate the input and output sizes
        self.input_size, self.output_size = 0, 0
        for reparameterizer in reparameterizer_list:
            self.input_size += reparameterizer.input_size
            self.output_size += reparameterizer.output_size

    def prior(self, batch_size):
        return [reparameterizer.prior(batch_size)
                for reparameterizer in self.reparmeterizers]

    def mutual_info(self, params_list):
        assert isinstance(params_list, list)
        mut_info_list = []
        current_params = 0
        for reparameterizer in self.reparmeterizers:
            if isinstance(reparameterizer, GumbelSoftmax):
                mut_info_list.append(reparameterizer.mutual_info(params_list[current_params]))
                current_params += 1

        return mut_info_list

    def log_likelihood(self, z_list, params_list):
        assert len(z_list) == len(params_list) == len(self.reparmeterizers)
        return [reparameterizer.log_likelihood(z_i, param_i)
                for reparameterizer, z_i, param_i in zip(self.reparmeterizers, z_list, params_list)]

    def reparmeterize(self, logits_list):
        assert len(logits_list) == len(self.reparmeterizers)
        ret_dict = {}
        for i, (reparameterizer, logits) in enumerate(zip(self.reparmeterizers, logits_list)):
            z, params = reparameterizer(logits)
            ret_dict['z_%d'%i] = z
            ret_dict['params_%d'%i] = params

        return ret_dict

    def kl(self, dist_a_list):
        return [reparameterizer.kl(dist_a_i)
                for reparameterizer, dist_a_i in zip(self.reparmeterizers, dist_a_list)]

    def forward(self, logits_list):
        return self.reparmeterize(logits_list)
