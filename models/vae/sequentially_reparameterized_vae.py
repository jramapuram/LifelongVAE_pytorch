from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from helpers.utils import float_type
from models.reparameterizers.gumbel import GumbelSoftmax
from models.reparameterizers.mixture import Mixture
from models.reparameterizers.isotropic_gaussian import IsotropicGaussian
from models.vae.abstract_vae import AbstractVAE


class StubReparameterizer(object):
    ''' A simple struct to hold the input and output size
        since other modules directly query input and output sizes '''
    def __init__(self, input_size, output_size, input_reparameterizer, config):
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.input_reparameterizer = input_reparameterizer

    def prior(self, batch_size):
        return self.input_reparameterizer.prior(batch_size)


class SequentiallyReparameterizedVAE(AbstractVAE):
    def __init__(self, input_shape, activation_fn=nn.ELU, num_current_model=0,
                 reparameterizer_strs=["discrete", "isotropic_gaussian"], **kwargs):
        super(SequentiallyReparameterizedVAE, self).__init__(input_shape,
                                                             activation_fn=activation_fn,
                                                             num_current_model=num_current_model,
                                                             **kwargs)

        # build the sequential set of reparameterizers
        self.reparameterizer_strs = reparameterizer_strs
        self.reparameterizers, self.reparameterizer \
            = self._build_sequential_reparameterizers(reparameterizer_strs)
        print(self.reparameterizers)

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def _build_sequential_reparameterizers(self, reparam_str_list):
        ''' helper to build all the reparameterizers '''
        reparameterizers = []
        encoder_output_size = None
        for reparam_str in reparam_str_list:
            if reparam_str == "isotropic_gaussian":
                print("adding isotropic gaussian reparameterizer")
                reparameterizers.append(IsotropicGaussian(self.config))
            elif reparam_str == "discrete":
                print("adding gumbel softmax reparameterizer")
                reparameterizers.append(GumbelSoftmax(self.config))
            elif reparam_str == "mixture":
                print("adding mixture reparameterizer")
                reparameterizers.append(Mixture(num_discrete=self.config['discrete_size'],
                                                num_continuous=self.config['continuous_size'],
                                                config=self.config))
            else:
                raise Exception("unknown reparameterization type")

            # the encoder projects to the first reparameterization's
            # input_size, so tabulate this to use in all the seq models below
            if encoder_output_size is None:
                encoder_output_size = reparameterizers[-1].input_size

            # tweak the reparameterizer to add a dense skip-network
            # XXX: parameterize the latent size
            reparameterizers[-1] = nn.Sequential(
                nn.Linear(encoder_output_size, 512),
                nn.BatchNorm1d(512),
                self.activation_fn(),
                nn.Linear(512, reparameterizers[-1].input_size),
                reparameterizers[-1]
            )

            if self.config['ngpu'] > 1:
                reparameterizers[-1] = nn.DataParallel(reparameterizers[-1])

            if self.config['cuda']:
                reparameterizers[-1].cuda()

        # a simple struct to hold input and output sizing
        # as well as the first reparameterizer in the chain
        r_container = StubReparameterizer(reparameterizers[0][-1].input_size,
                                          reparameterizers[-1][-1].output_size,
                                          reparameterizers[0][-1],
                                          self.config)
        return reparameterizers, r_container

    def get_name(self):
        param_str = ""
        for reparam_str in self.reparameterizer_strs:
            if reparam_str == "mixture":
                param_str += "mixturecat{}gauss{}_".format(
                    str(self.config['discrete_size']),
                    str(self.config['continuous_size'])
                )
            elif reparam_str == "isotropic_gaussian":
                param_str += "gauss{}_".format(str(self.config['continuous_size']))
            elif reparam_str == "discrete":
                param_str += "cat{}_".format(str(self.config['discrete_size']))

        return 'seqvae_' + super(SequentiallyReparameterizedVAE, self).get_name(param_str)

    def has_discrete(self):
        ''' True is our FIRST parameterize is discrete '''
        return isinstance(self.reparameterizers[0][-1], GumbelSoftmax)

    def get_reparameterizer_scalars(self):
        ''' basically returns tau from reparameterizers for now '''
        reparam_scalar_map = {}
        for i, reparam in enumerate(self.reparameterizers):
            if isinstance(reparam[-1], GumbelSoftmax):
                reparam_scalar_map['tau%d_scalar'%i] = reparam[-1].tau
            elif isinstance(reparam[-1], Mixture):
                reparam_scalar_map['tau%d_scalar'%i] = reparam[-1].discrete.tau

        return reparam_scalar_map

    def reparameterize(self, z):
        ''' reparameterize the latent logits appropriately '''
        batch_size = z.size(0)
        params_map = {}
        z_logits = z.clone().view(batch_size, -1)
        for i, reparameterizer in enumerate(self.reparameterizers):
            if i > 0: # add a residual connection
                self._lazy_init_dense(z_logits.size(-1), z.size(-1),
                                      name="residual_%d"%i)
                z = z + getattr(self, "residual_%d"%i)(z_logits)

            z, params = reparameterizer(z.contiguous().view(batch_size, -1))
            params_map['z_%d'%i] = z
            params_map['params_%d'%i] = params

        return z, params_map

    def decode(self, z):
        '''returns logits '''
        if self.config['use_relational_encoder']:
            # build a relational net as the encoder projection
            self._lazy_init_relational(self.reparameterizer.input_size,
                                       name='dec_proj')
        else:
            # project via linear layer [if necessary!]
            z_output_shp = int(np.prod(z.size()[1:]))
            self._lazy_init_dense(z_output_shp,
                                  self.reparameterizer.output_size,
                                  name='dec_proj')

        # project via decoder
        logits = self.decoder(self.dec_proj(z.contiguous()))
        return self._project_decoder_for_variance(logits)

    def posterior(self, x):
        ''' helper that does encode --> reparameterize '''
        z_logits = self.encoder(x)
        return self.reparameterize(z_logits)

    def forward(self, x):
        ''' params is a map of the latent variable's parameters'''
        z, params_map = self.posterior(x)
        return self.decode(z), params_map

    def generate(self, z):
        ''' given z_0 generate z_1, ... -> decode(z_last) '''
        for reparameterizer in self.reparameterizers[1:]:
            z, _ = reparameterizer(z)

        return self.decode(z)

    def kld(self, dists):
        ''' does the KL divergence between the posterior and the prior '''
        batch_size = dists['z_0'].size(0)
        kl = Variable(float_type(self.config['cuda'])(batch_size).zero_())
        for i, reparameterizer in enumerate(self.reparameterizers):
            kl += reparameterizer[-1].kl(dists['params_%d'%i])

        return kl

    def loss_function(self, recon_x, x, params):
        return super(SequentiallyReparameterizedVAE, self).loss_function(recon_x, x, params)
