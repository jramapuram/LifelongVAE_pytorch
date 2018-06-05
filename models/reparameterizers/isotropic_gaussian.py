from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
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

    def prior(self, batch_size, **kwargs):
        scale_var = 1.0 if 'scale_var' not in kwargs else kwargs['scale_var']
        return Variable(
            float_type(self.config['cuda'])(batch_size, self.output_size).normal_(mean=0, std=scale_var)
        )

    def _reparametrize_gaussian(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = float_type(self.config['cuda'])(std.size()).normal_()
            eps = Variable(eps)
            z = eps.mul(std).add_(mu)
            return z, {'z': z, 'mu': mu, 'logvar': logvar}

        return mu, {'z': mu, 'mu': mu, 'logvar': logvar}

    def reparmeterize(self, logits, eps=1e-9):
        feature_size = logits.size(-1)
        assert feature_size % 2 == 0 and feature_size // 2 == self.output_size
        mu = logits[:, 0:int(feature_size/2)]
        sigma = logits[:, int(feature_size/2):] + eps
        # sigma = F.softplus(logits[:, int(feature_size/2):]) + eps
        # sigma = F.hardtanh(logits[:, int(feature_size/2):], min_val=-6.,max_val=2.)
        return self._reparametrize_gaussian(mu, sigma)

    @staticmethod
    def cross_entropy_kl_version(dist_a, dist_b):
        # KL = CE - E, thus CE = KL + E
        return dist_a.entropy() + D.kl_divergence(dist_a, dist_b)

    def mutual_info(self, params, eps=1e-9):
        if self.config['monte_carlo_infogain']:
            return self.mutual_info_monte_carlo(params, eps)

        return self.mutual_info_analytic(params, eps)

    def mutual_info_monte_carlo(self, params, eps=1e-9):
        # I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)
        q_z_given_x, _ = self._reparametrize_gaussian(params['q_z_given_xhat']['gaussian']['mu'],
                                                      params['q_z_given_xhat']['gaussian']['logvar'])
        log_q_z_given_x = torch.log(torch.abs(q_z_given_x) + eps)
        p_z = self.prior(log_q_z_given_x.size()[0])
        crossent_loss = -torch.sum(log_q_z_given_x * p_z, dim=1)
        ent_loss = -torch.sum(torch.log(torch.abs(p_z) + eps) * p_z, dim=1)
        return crossent_loss + ent_loss

    def mutual_info_analytic(self, params, eps=1e-9):
        # I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)
        z_true = D.Normal(params['gaussian']['mu'],
                          params['gaussian']['logvar'])
        z_match = D.Normal(params['q_z_given_xhat']['gaussian']['mu'],
                           params['q_z_given_xhat']['gaussian']['logvar'])
        kl_proxy_to_xent = torch.sum(D.kl_divergence(z_match, z_true), dim=-1)
        return  kl_proxy_to_xent

    # def mutual_info_analytic(self, params, eps=1e-9):
    #     # I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)
    #     z_0_1 = D.Normal(zeros_like(params['gaussian']['mu']),
    #                      ones_like(params['gaussian']['logvar']))
    #     z_c = D.Normal(params['q_z_given_xhat']['gaussian']['mu'],
    #                    params['q_z_given_xhat']['gaussian']['logvar'])
    #     # z_c = D.Normal(params['gaussian']['mu'],
    #     #                params['gaussian']['logvar'])
    #     crossent_loss = torch.sum(self.cross_entropy_kl_version(z_0_1, z_c), dim=-1)
    #     ent_loss = torch.sum(z_0_1.entropy(), dim=-1)
    #     return  ent_loss + crossent_loss

    @staticmethod
    def _kld_gaussian_N_0_1(mu, logvar):
        standard_normal = D.Normal(zeros_like(mu), ones_like(logvar))
        normal = D.Normal(mu, logvar)
        return torch.sum(D.kl_divergence(normal, standard_normal), -1)
        # return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)

    def kl(self, dist_a):
        return IsotropicGaussian._kld_gaussian_N_0_1(
            dist_a['gaussian']['mu'], dist_a['gaussian']['logvar']
        )

    def log_likelihood(self, z, params):
        return D.Normal(params['gaussian']['mu'],
                        params['gaussian']['logvar']).log_prob(z)

    def forward(self, logits):
        z, gauss_params = self.reparmeterize(logits)
        return z, { 'z': z, 'gaussian':  gauss_params }
