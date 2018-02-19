from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
from collections import OrderedDict

from models.layers import View, Identity, UpsampleConvLayer, flatten_layers
from models.gumbel import GumbelSoftmax
from models.mixture import Mixture
from models.isotropic_gaussian import IsotropicGaussian
from helpers.utils import float_type, ones_like


def log_logistic_256(x, mean, log_s, average=False, dim=None):
    binsize = 1. / 256.
    scale = torch.exp(log_s)
    # make sure image fit proper values
    x = torch.floor(x/binsize) * binsize
    # calculate normalized values for a bin
    x_plus = (x + binsize - mean) / scale
    x_minus = (x - mean) / scale
    # calculate logistic CDF for a bin
    cdf_plus = F.sigmoid(x_plus)
    cdf_minus = F.sigmoid(x_minus)
    # calculate final log-likelihood for an image
    log_logistic_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if average:
        return torch.mean( log_logistic_256, dim )

    return torch.sum( log_logistic_256, dim )


class VAE(nn.Module):
    def __init__(self, input_shape, activation_fn=nn.ELU, num_current_model=0, **kwargs):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.num_current_model = num_current_model
        self.activation_fn = activation_fn
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1

        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        # build the reparameterizer
        if self.config['reparam_type'] == "isotropic_gaussian":
            print("using isotropic gaussian reparameterizer")
            self.reparameterizer = IsotropicGaussian(self.config)
        elif self.config['reparam_type'] == "discrete":
            print("using gumbel softmax reparameterizer")
            self.reparameterizer = GumbelSoftmax(self.config)
        elif self.config['reparam_type'] == "mixture":
            print("using mixture reparameterizer")
            self.reparameterizer = Mixture(num_discrete=self.config['discrete_size'],
                                           num_continuous=self.config['continuous_size'],
                                           config=self.config)
        else:
            raise Exception("unknown reparameterization type")

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.full_model = None

    def get_name(self):
        if self.config['reparam_type'] == "mixture":
            param_str = "_disc" + str(self.config['discrete_size']) + \
                        "_cont" + str(self.config['continuous_size'])
        elif self.config['reparam_type'] == "isotropic_gaussian":
            param_str = "_cont" + str(self.config['continuous_size'])
        elif self.config['reparam_type'] == "discrete":
            param_str = "_disc" + str(self.config['discrete_size'])

        full_hash_str = "_" + str(self.config['layer_type']) + \
                        "_input" + str(self.input_shape) + \
                        param_str + \
                        "_batch" + str(self.config['batch_size']) + \
                        "_mut" + str(self.config['mut_reg']) + \
                        "_filter_depth" + str(self.config['filter_depth']) + \
                        "_nll" + str(self.config['nll_type']) + \
                        "_reparam" + str(self.config['reparam_type']) + \
                        "_lr" + str(self.config['lr']) + \
                        "_ngpu" + str(self.config['ngpu'])

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
        return 'vae_' + '_'.join(self.config['task']) + full_hash_str

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
                self.activation_fn(inplace=True),
                # state dim: 32 x 28 x 28
                nn.Conv2d(self.config['filter_depth'], self.config['filter_depth']*2, 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*2),
                self.activation_fn(inplace=True),
                # state dim: 64 x 13 x 13
                nn.Conv2d(self.config['filter_depth']*2, self.config['filter_depth']*4, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*4),
                self.activation_fn(inplace=True),
                # state dim: 128 x 10 x 10
                nn.Conv2d(self.config['filter_depth']*4, self.config['filter_depth']*8, 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*8),
                self.activation_fn(inplace=True),
                # state dim: 256 x 4 x 4
                nn.Conv2d(self.config['filter_depth']*8, self.config['filter_depth']*16, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*16),
                self.activation_fn(inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(self.config['filter_depth']*16, self.config['filter_depth']*16, 1, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*16),
                self.activation_fn(inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(self.config['filter_depth']*16, self.reparameterizer.input_size, 1, stride=1, bias=True),
                nn.BatchNorm2d(self.reparameterizer.input_size),
                self.activation_fn(inplace=True)
                # output dim: opt.z_dim x 1 x 1
            )
        elif self.config['layer_type'] == 'dense':
            encoder = nn.Sequential(
                View([-1, int(np.prod(self.input_shape))]),
                nn.Linear(int(np.prod(self.input_shape)), 512),
                nn.BatchNorm1d(512),
                self.activation_fn(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                self.activation_fn(),
                nn.Linear(512, self.reparameterizer.input_size)
                # nn.BatchNorm1d(self.reparameterizer.input_size),
                # self.activation_fn(),
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
                View([-1, self.reparameterizer.output_size, 1, 1]),
                # input dim: z_dim x 1 x 1
                nn.ConvTranspose2d(self.reparameterizer.output_size, self.config['filter_depth']*8, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*8),
                self.activation_fn(inplace=True),
                # state dim:   256 x 4 x 4
                nn.ConvTranspose2d(self.config['filter_depth']*8, self.config['filter_depth']*4, 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*4),
                self.activation_fn(inplace=True),
                # state dim: 128 x 10 x 10
                nn.ConvTranspose2d(self.config['filter_depth']*4, self.config['filter_depth']*2, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*2),
                self.activation_fn(inplace=True),
                # state dim: 64 x 13 x 13
                nn.ConvTranspose2d(self.config['filter_depth']*2, self.config['filter_depth'], 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']),
                self.activation_fn(inplace=True),
                # state dim: 32 x 28 x 28
                nn.ConvTranspose2d(self.config['filter_depth'], self.config['filter_depth'], 5, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']),
                self.activation_fn(inplace=True),
                # state dim: 32 x 32 x 32
                nn.Conv2d(self.config['filter_depth'], self.chans, 1, stride=1, bias=True),
                # output dim: num_channels x 32 x 32
                upsampler if self.input_shape[1:] != bilinear_size else Identity()
            )
        elif self.config['layer_type'] == 'dense':
            decoder = nn.Sequential(
                View([-1, self.reparameterizer.output_size]),
                nn.Linear(self.reparameterizer.output_size, 512),
                nn.BatchNorm1d(512),
                self.activation_fn(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                self.activation_fn(),
                nn.Linear(512, int(np.prod(self.input_shape))),
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
                # nn.BatchNorm1d(input_size),
                # self.activation_fn(),
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

    def nll_activation(self, logits):
        if self.config['nll_type'] == "clamp":
            num_half_chans = logits.size(1) // 2
            logits_mu = logits[:, 0:num_half_chans, :, :]
            return torch.clamp(logits_mu, min=0.+1./512., max=1.-1./512.)
        elif self.config['nll_type'] == "gaussian":
            num_half_chans = logits.size(1) // 2
            logits_mu = logits[:, 0:num_half_chans, :, :]
            #return F.sigmoid(logits_mu)
            return logits_mu
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

    def _project_decoder_for_variance(self, logits):
        ''' if we have a nll with variance
            then project it to the required dimensions '''
        if self.config['nll_type'] == 'gaussian' \
           or self.config['nll_type'] == 'clamp':
            if not hasattr(self, 'decoder_projector'):
                if self.config['layer_type'] == 'conv':
                    self.decoder_projector = nn.Sequential(
                        nn.BatchNorm2d(self.chans),
                        self.activation_fn(inplace=True),
                        nn.ConvTranspose2d(self.chans, self.chans*2, 1, stride=1, bias=False)
                    )
                else:
                    input_flat = int(np.prod(self.input_shape))
                    self.decoder_projector = nn.Sequential(
                        View([-1, input_flat]),
                        nn.BatchNorm1d(input_flat),
                        self.activation_fn(inplace=True),
                        nn.Linear(input_flat, input_flat*2, bias=True),
                        View([-1, self.chans*2, *self.input_shape[1:]])
                    )

                if self.config['cuda']:
                    self.decoder_projector.cuda()

            return self.decoder_projector(logits)

        # bernoulli reconstruction
        return logits

    def decode(self, z):
        '''returns logits '''
        logits = self.decoder(z.contiguous())
        return self._project_decoder_for_variance(logits)

    def compile_full_model(self):
        '''NOTE: add decoder projection'''
        if hasattr(self, 'enc_proj'):
            if not self.full_model:
                full_model_list, _ = flatten_layers(
                    nn.Sequential(
                        self.encoder,
                        self.enc_proj,
                        self.reparameterizer,
                        self.decoder
                    ))
                self.full_model = nn.Sequential(OrderedDict(full_model_list))
        else:
            raise Exception("cant compile full model till you lazy-init the dense layer")

    def forward(self, x):
        ''' params is a map of the latent variable's parameters'''
        z_logits = self.encode(x)
        z, params = self.reparameterize(z_logits)
        return self.decode(z), params

    def nll(self, recon_x, x):
        nll_map = {
            "gaussian": self.nll_gaussian,
            "bernoulli": self.nll_bernoulli,
            "clamp": self.nll_clamp
        }
        return nll_map[self.config['nll_type']](x, recon_x)

    def nll_bernoulli(self, x, recon_x_logits):
        batch_size = x.size(0)
        nll = D.Bernoulli(logits=recon_x_logits.view(batch_size, -1)).log_prob(
            x.view(batch_size, -1)
        )
        return -torch.sum(nll, dim=-1)

    def nll_clamp(self, x, recon):
        batch_size, num_half_chans = x.size(0), recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans, :, :]
        recon_logvar = recon[:, num_half_chans:, :, :]

        return log_logistic_256(x.view(batch_size, -1),
                                torch.clamp(recon_mu.view(batch_size, -1), min=0.+1./512., max=1.-1./512.),
                                F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0),
                                dim=-1)

    def nll_laplace(self, x, recon):
        batch_size, num_half_chans = x.size(0), recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans, :, :]
        recon_logvar = recon[:, num_half_chans:, :, :]
        #recon_logvar = ones_like(recon_mu, self.config['cuda'])

        nll = D.Laplace(
            # recon_mu.view(batch_size, -1),
            recon_mu.view(batch_size, -1),
            # F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
            recon_logvar.view(batch_size, -1)
        ).log_prob(x.view(batch_size, -1))
        return -torch.sum(nll, dim=-1)

    def nll_gaussian(self, x, recon):
        batch_size, num_half_chans = x.size(0), recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans, :, :]
        recon_logvar = recon[:, num_half_chans:, :, :]

        # XXX: currently broken, so set var to 1
        recon_logvar = ones_like(recon_mu, self.config['cuda'])

        nll = D.Normal(
            recon_mu.view(batch_size, -1),
            #F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
            recon_logvar.view(batch_size, -1)
        ).log_prob(x.view(batch_size, -1))
        return -torch.sum(nll, dim=-1)

    def kld(self, dist_a):
        ''' accepts param maps for dist_a and dist_b,
            TODO: make generic and accept two distributions'''
        return torch.sum(self.reparameterizer.kl(dist_a), dim=-1)

    def loss_function(self, recon_x, x, params):
        # tf: elbo = -log_likelihood + latent_kl
        # tf: cost = elbo + consistency_kl - self.mutual_info_reg * mutual_info_regularizer
        nll = self.nll(recon_x, x)
        kld = self.kld(params)
        elbo = nll + kld
        mut_info = Variable(
            float_type(self.config['cuda'])(x.size(0)).zero_()
        )

        # add the mutual information regularizer if
        # running a mixture model ONLY
        if self.config['reparam_type'] == 'mixture' \
           or self.config['reparam_type'] == 'discrete'\
           and not self.config['disable_regularizers']:
            mut_info = self.reparameterizer.mutual_info(params)
            # print("torch.norm(kld, p=2)", torch.norm(kld, p=2))
            # mut_info = torch.clamp(mut_info, min=0, max=torch.norm(kld, p=2).data[0])
            mut_info = mut_info / torch.norm(mut_info, p=2)

        loss = elbo - self.config['mut_reg'] * mut_info
        return {
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(kld),
            'mut_info_mean': torch.mean(mut_info)
        }
