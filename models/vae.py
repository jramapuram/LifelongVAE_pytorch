from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.layers import View, Identity, UpsampleConvLayer
from models.gumbel import GumbelSoftmax
from models.mixture import Mixture
from models.isotropic_gaussian import IsotropicGaussian
from helpers.utils import float_type


class VAE(nn.Module):
    def __init__(self, input_shape, latent_size, activation_fn=nn.ReLU, **kwargs):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.activation_fn = activation_fn
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1

        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # build the reparameterizer
        if self.config['reparam_type'] == "isotropic_gaussian":
            print("using isotropic gaussian reparameterizer")
            self.reparameterizer = IsotropicGaussian(self.config)
        elif self.config['reparam_type'] == "discrete":
            print("using gumbel softmax reparameterizer")
            self.reparameterizer = GumbelSoftmax(self.config)
        elif self.config['reparam_type'] == "mixture":
            print("using mixture reparameterizer")
            assert self.config['latent_size'] > self.config['mixture_discrete_size']
            #num_continuous = self.config['latent_size'] - self.config['mixture_discrete_size']
            self.reparameterizer = Mixture(num_discrete=self.config['mixture_discrete_size'],
                                           #num_continuous=num_continuous,
                                           num_continuous=self.config['latent_size'],
                                           config=self.config)
        else:
            raise Exception("unknown reparameterization type")

    def get_name(self):
        full_hash_str = "_input" + str(self.input_shape) + \
                        "_latent" + str(self.config['latent_size']) + \
                        "_batch" + str(self.config['batch_size']) + \
                        "_filter_depth" + str(self.config['filter_depth']) + \
                        "_nll" + str(self.config['nll_type']) + \
                        "_reparam" + str(self.config['reparam_type']) + \
                        "_lr" + str(self.config['lr'])

        full_hash_str = full_hash_str.strip().lower().replace('[', '')  \
                                                     .replace(']', '')  \
                                                     .replace(' ', '')  \
                                                     .replace('{', '') \
                                                     .replace('}', '') \
                                                     .replace(',', '_') \
                                                     .replace(':', '') \
                                                     .replace('(', '') \
                                                     .replace(')', '') \
                                                     .replace('\'', '')
        return 'supvae_' + self.config['task'] + full_hash_str

    def build_encoder(self):
        ''' helper function to build convolutional encoder'''

        if self.config['layer_type'] == 'conv':
            # build an upsampler (possible downsampler in some cases) to 32x32
            bilinear_size = [32, 32]  # XXX: hard coded
            upsampler = nn.Upsample(size=bilinear_size, mode='bilinear')
            encoder = nn.Sequential(
                upsampler if self.input_shape[1:] != bilinear_size else Identity(),
                # input dim: num_channels x 32 x 32
                nn.Conv2d(self.chans, self.config['filter_depth'], 5, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']),
                nn.ELU(inplace=True),
                # state dim: 32 x 28 x 28
                nn.Conv2d(self.config['filter_depth'], self.config['filter_depth']*2, 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*2),
                nn.ELU(inplace=True),
                # state dim: 64 x 13 x 13
                nn.Conv2d(self.config['filter_depth']*2, self.config['filter_depth']*4, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*4),
                nn.ELU(inplace=True),
                # state dim: 128 x 10 x 10
                nn.Conv2d(self.config['filter_depth']*4, self.config['filter_depth']*8, 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*8),
                nn.ELU(inplace=True),
                # state dim: 256 x 4 x 4
                nn.Conv2d(self.config['filter_depth']*8, self.config['filter_depth']*16, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*16),
                nn.ELU(inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(self.config['filter_depth']*16, self.config['filter_depth']*16, 1, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*16),
                nn.ELU(inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(self.config['filter_depth']*16, self.latent_size, 1, stride=1, bias=True)
                # output dim: opt.z_dim x 1 x 1
            )
        elif self.config['layer_type'] == 'dense':
            encoder = nn.Sequential(
                View([-1, int(np.prod(self.input_shape))]),
                nn.Linear(int(np.prod(self.input_shape)), self.latent_size),
                nn.BatchNorm1d(self.latent_size),
                nn.ReLU(),
                nn.Linear(self.latent_size, self.latent_size),
                nn.BatchNorm1d(self.latent_size),
                nn.ReLU(),
                nn.Linear(self.latent_size, self.latent_size),
                nn.BatchNorm1d(self.latent_size),
                nn.ReLU(),
            )
        else:
            raise Exception("unknown layer type requested")

        if self.config['ngpu'] > 1:
            encoder = nn.DataParallel(encoder)

        if self.config['cuda']:
            encoder = encoder.cuda()

        return encoder

    def build_decoder(self):
        ''' helper function to build convolutional decoder'''
        bilinear_size = [32, 32]  # XXX: hard coded
        if self.config['layer_type'] == 'conv':
            upsampler = nn.Upsample(size=self.input_shape[1:], mode='bilinear')
            decoder = nn.Sequential(
                View([-1, self.latent_size, 1, 1]),
                # input dim: z_dim x 1 x 1
                nn.ConvTranspose2d(self.latent_size, self.config['filter_depth']*8, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*8),
                nn.ELU(inplace=True),
                # state dim:   256 x 4 x 4
                nn.ConvTranspose2d(self.config['filter_depth']*8, self.config['filter_depth']*4, 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*4),
                nn.ELU(inplace=True),
                # state dim: 128 x 10 x 10
                nn.ConvTranspose2d(self.config['filter_depth']*4, self.config['filter_depth']*2, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*2),
                nn.ELU(inplace=True),
                # state dim: 64 x 13 x 13
                nn.ConvTranspose2d(self.config['filter_depth']*2, self.config['filter_depth'], 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']),
                nn.ELU(inplace=True),
                # state dim: 32 x 28 x 28
                nn.ConvTranspose2d(self.config['filter_depth'], self.config['filter_depth'], 5, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']),
                nn.ELU(inplace=True),
                # state dim: 32 x 32 x 32
                nn.Conv2d(self.config['filter_depth'], self.chans, 1, stride=1, bias=True),
                # output dim: num_channels x 32 x 32
                upsampler if self.input_shape[1:] != bilinear_size else Identity()
            )
        elif self.config['layer_type'] == 'dense':
            decoder = nn.Sequential(
                View([-1, self.latent_size]),
                nn.Linear(self.latent_size, self.latent_size),
                nn.BatchNorm1d(self.latent_size),
                nn.ReLU(),
                nn.Linear(self.latent_size, self.latent_size),
                nn.BatchNorm1d(self.latent_size),
                nn.ReLU(),
                nn.Linear(self.latent_size, int(np.prod(self.input_shape))),
                View([-1] + self.input_shape)
            )
        else:
            raise Exception("unknown layer type requested")

        if self.config['ngpu'] > 1:
            decoder = nn.DataParallel(decoder)

        if self.config['cuda']:
            decoder = decoder.cuda()

        return decoder

    def _lazy_init_dense(self, input_size, output_size, name='enc_proj'):
        '''initialize the dense linear projection lazily
           because determining convolutional output size
           is annoying '''
        if not hasattr(self, name):
            # build a simple linear projector
            setattr(self, name, nn.Sequential(
                View([-1, input_size]),
                nn.Linear(input_size, output_size)
            ))

            if self.config['ngpu'] > 1:
                setattr(self, name,
                        nn.DataParallel(getattr(self, name))
                )

            if self.config['cuda']:
                setattr(self, name, getattr(self, name).cuda())

    def reparameterize(self, logits):
        ''' reparameterizes the latent logits appropriately '''
        return self.reparameterizer(logits)

    def _nll_activation(self, logits):
        if self.config['nll_type'] == "gaussian":
            return logits
        elif self.config['nll_type'] == "bernoulli":
            return F.sigmoid(logits)
        else:
            raise Exception("unknown nll provided")

    def encode(self, x):
        ''' encodes via a convolution
            and lazy init's a dense projector'''
        conv = self.encoder(x)         # do the convolution
        conv_output_shp = int(np.prod(conv.size()[1:]))

        # project via linear layer
        self._lazy_init_dense(conv_output_shp,
                              self.reparameterizer.input_size,
                              name='enc_proj')
        return self.enc_proj(conv)

    def decode(self, z):
        # project via linear layer
        self._lazy_init_dense(self.reparameterizer.output_size,
                              self.latent_size, 'dec_proj')
        z_proj = self.dec_proj(z)

        logits = self.decoder(z_proj)
        return self._nll_activation(logits)

    def forward(self, x):
        ''' params is a map of the latent variable's parameters'''
        z_logits = self.encode(x)
        z, params = self.reparameterize(z_logits)
        return self.decode(z), params

    def nll(self, recon_x, x, params):
        if self.config['nll_type'] == "gaussian":
            return self.nll_gaussian(recon_x, x,
                                     params['mu'],
                                     params['logvar'])
        elif self.config['nll_type'] == "bernoulli":
            return self.nll_bernoulli(recon_x, x)
        else:
            raise Exception("Unknown NLL")

    def nll_bernoulli(self, recon_x_logits, x, size_average=True):
        return F.binary_cross_entropy_with_logits(recon_x_logits, x,
                                                  size_average=size_average)

    def nll_gaussian(self, recon_x, x, mu, logvar):
        # Helpers to get the gaussian log-likelihood
        # pulled from tensorflow
        # (https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/distributions/normal.py)
        def _z(self, x, loc, scale, eps=1e-9):
            """Standardize input `x` to a unit normal."""
            return (x - loc) / (scale + eps)

        def _log_unnormalized_prob(self, x, loc, scale):
            return -0.5 * torch.pow(self._z(x, loc, scale), 2)

        def _log_normalization(self, scale, eps=1e-9):
            return torch.log(scale + eps) + 0.5 * np.log(2. * np.pi)

        return self._log_unnormalized_prob(x, mu, logvar) \
            - self._log_normalization(logvar)

    def kld(self, dist_a):
        ''' accepts param maps for dist_a and dist_b,
            TODO: make generic and accept two distributions'''
        return self.reparameterizer.kl(dist_a)

    def loss_function(self, recon_x, x, params):
        nll = self.nll(recon_x, x, params)
        kld = self.kld(params)
        mut_info = 0.0

        # add the mutual information regularizer if
        # running a mixture model ONLY
        if self.config['reparam_type'] == 'mixture':
            mut_info += self.reparameterizer.mutual_info(params)

        return {
            'loss': nll + kld + self.config['mut_reg'] * mut_info,
            'nll': nll,
            'kld': kld,
            'mut_info': mut_info,
        }
