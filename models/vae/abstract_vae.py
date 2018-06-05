from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict, Counter

from helpers.utils import float_type
from models.relational_network import RelationalNetwork
from helpers.layers import View, flatten_layers, Identity, \
    build_gated_conv_encoder, build_conv_encoder, build_dense_encoder, build_relational_conv_encoder, \
    build_gated_conv_decoder, build_conv_decoder, build_dense_decoder, build_pixelcnn_decoder, str_to_activ_module
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn


class AbstractVAE(nn.Module):
    ''' abstract base class for VAE, both sequentialVAE and parallelVAE inherit this '''
    def __init__(self, input_shape, **kwargs):
        super(AbstractVAE, self).__init__()
        self.input_shape = input_shape
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1

        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        # grab the activation nn.Module from the string
        self.activation_fn = str_to_activ_module(self.config['activation'])

        # placeholder in order to sequentialize model
        self.full_model = None

    def get_name(self, reparam_str):
        ''' helper to get the name of the model '''
        es_str = "es" + str(int(self.config['early_stop'])) if self.config['early_stop'] \
                 else "epochs" + str(self.config['epochs'])
        full_hash_str = "_{}_{}act{}_da{}_st{}{}_dr{}_re{}_pd{}_klr{}_gsv{}_mcig{}_mcs{}{}_input{}_batch{}_mut{}d{}c_filter{}_nll{}_lr{}_{}_ngpu{}".format(
            str(self.config['layer_type']),
            reparam_str,
            str(self.activation_fn.__name__),
            str(int(self.config['disable_augmentation'])),
            str(int(not self.config['disable_student_teacher'])),
            "_ewc{}".format(str(int(self.config['ewc_gamma']))) if int(self.config['ewc_gamma'] > 0) else "",
            str(int(self.config['disable_regularizers'])),
            str(int(self.config['use_relational_encoder'])),
            str(int(self.config['use_pixel_cnn_decoder'])),
            str(self.config['kl_reg']),
            str(self.config['generative_scale_var']),
            str(int(self.config['monte_carlo_infogain'])),
            str(self.config['mut_clamp_strategy']),
            "{}".format(str(self.config['mut_clamp_value'])) if self.config['mut_clamp_strategy'] == 'clamp' else "",
            str(self.input_shape),
            str(self.config['batch_size']),
            str(self.config['discrete_mut_info']),
            str(self.config['continuous_mut_info']),
            str(self.config['filter_depth']),
            str(self.config['nll_type']),
            str(self.config['lr']),
            es_str,
            str(self.config['ngpu'])
        )
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
        task_cleaned = AbstractVAE._clean_task_str(self.config['task'])
        return task_cleaned + full_hash_str


    @staticmethod
    def _clean_task_str(task_str):
        ''' helper to reduce string length.
            eg: mnist+svhn+mnist --> mnist2svhn1 '''
        result_str = ''
        if '+' in task_str:
            splits = Counter(task_str.split('+'))
            for k, v in splits.items():
                result_str += '{}{}'.format(k, v)

            return result_str

        return task_str

    def build_encoder(self):
        ''' helper function to build convolutional or dense encoder '''
        if self.config['layer_type'] == 'conv':
            if self.config['use_relational_encoder']:
                encoder = build_relational_conv_encoder(input_shape=self.input_shape,
                                                        filter_depth=self.config['filter_depth'],
                                                        activation_fn=self.activation_fn)
                raise NotImplementedError
            else:
                conv_builder = build_gated_conv_encoder \
                           if self.config['disable_gated_conv'] is False else build_conv_encoder
                encoder = conv_builder(input_shape=self.input_shape,
                                       output_size=self.reparameterizer.input_size,
                                       filter_depth=self.config['filter_depth'],
                                       activation_fn=self.activation_fn,
                                       normalization_str=self.config['normalization'])
        elif self.config['layer_type'] == 'dense':
            encoder = build_dense_encoder(input_shape=self.input_shape,
                                          output_size=self.reparameterizer.input_size,
                                          latent_size=512,
                                          activation_fn=self.activation_fn,
                                          normalization_str=self.config['normalization'])
        else:
            raise Exception("unknown layer type requested")

        if self.config['ngpu'] > 1:
            encoder = nn.DataParallel(encoder)

        if self.config['cuda']:
            encoder = encoder.cuda()

        return encoder

    def build_decoder(self):
        ''' helper function to build convolutional or dense decoder'''
        if self.config['layer_type'] == 'conv':
            conv_builder = build_gated_conv_decoder \
                           if self.config['disable_gated_conv'] is False else build_conv_decoder
            decoder = nn.Sequential(
                conv_builder(input_size=self.reparameterizer.output_size,
                             output_shape=self.input_shape,
                             filter_depth=self.config['filter_depth'],
                             activation_fn=self.activation_fn,
                             normalization_str=self.config['normalization'])
            )
            if self.config['use_pixel_cnn_decoder']:
                print("adding pixel CNN decoder...")
                decoder = nn.Sequential(
                    decoder,
                    build_pixelcnn_decoder(input_size=self.chans,
                                           output_shape=self.input_shape,
                                           normalization_str=self.config['normalization'])
                )

        elif self.config['layer_type'] == 'dense':
            decoder = build_dense_decoder(input_size=self.reparameterizer.output_size,
                                          output_shape=self.input_shape,
                                          activation_fn=self.activation_fn,
                                          normalization_str=self.config['normalization'])
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

    def _lazy_init_relational(self, output_size, name='enc_proj'):
        '''initialize a relational network lazily
           because determining convolutional output size
           is annoying '''
        if not hasattr(self, name):
            setattr(self, name, RelationalNetwork(hidden_size=512, #XXX
                                                  output_size=output_size,
                                                  cuda=self.config['cuda'],
                                                  ngpu=self.config['ngpu']))

            if self.config['ngpu'] > 1:
                setattr(self, name,
                        nn.DataParallel(getattr(self, name))
                )

            if self.config['cuda']:
                setattr(self, name, getattr(self, name).cuda())

    def _project_decoder_for_variance(self, logits):
        ''' if we have a nll with variance
            then project it to the required dimensions '''
        if self.config['nll_type'] == 'gaussian' \
           or self.config['nll_type'] == 'clamp':
            if not hasattr(self, 'decoder_projector'):
                if self.config['layer_type'] == 'conv':
                    self.decoder_projector = nn.Sequential(
                        nn.BatchNorm2d(self.chans) if not self.config['disable_batchnorm'] else Identity(),
                        self.activation_fn(inplace=True),
                        nn.ConvTranspose2d(self.chans, self.chans*2, 1, stride=1, bias=False)
                    )
                else:
                    input_flat = int(np.prod(self.input_shape))
                    self.decoder_projector = nn.Sequential(
                        View([-1, input_flat]),
                        nn.BatchNorm1d(input_flat) if not self.config['disable_batchnorm'] else Identity(),
                        self.activation_fn(inplace=True),
                        nn.Linear(input_flat, input_flat*2, bias=True),
                        View([-1, self.chans*2, *self.input_shape[1:]])
                    )

                if self.config['cuda']:
                    self.decoder_projector.cuda()

            return self.decoder_projector(logits)

        # bernoulli reconstruction
        return logits

    def compile_full_model(self):
        ''' takes all the submodules and module-lists
            and returns one gigantic sequential_model '''
        if not self.full_model:
            full_model_list, _ = flatten_layers(self)
            self.full_model = nn.Sequential(OrderedDict(full_model_list))

    def nll_activation(self, logits):
        return nll_activation_fn(logits, self.config['nll_type'])

    def forward(self, x):
        ''' params is a map of the latent variable's parameters'''
        z, params = self.posterior(x)
        return self.decode(z), params

    def loss_function(self, recon_x, x, params, mut_info=None):
        # tf: elbo = -log_likelihood + latent_kl
        # tf: cost = elbo + consistency_kl - self.mutual_info_reg * mutual_info_regularizer
        nll = nll_fn(x, recon_x, self.config['nll_type'])
        kld = self.config['kl_reg'] * self.kld(params)
        elbo = nll + kld

        # handle the mutual information term
        if mut_info is None:
            mut_info = Variable(
                float_type(self.config['cuda'])(x.size(0)).zero_()
            )
        else:
            # Clamping strategies
            mut_clamp_strategy_map = {
                'none': lambda mut_info: mut_info,
                'norm': lambda mut_info: mut_info / torch.norm(mut_info, p=2),
                'clamp': lambda mut_info: torch.clamp(mut_info,
                                                      min=-self.config['mut_clamp_value'],
                                                      max=self.config['mut_clamp_value'])
            }
            mut_info = mut_clamp_strategy_map[self.config['mut_clamp_strategy'].strip().lower()](mut_info)

        loss = elbo - mut_info
        return {
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(kld),
            'mut_info_mean': torch.mean(mut_info)
        }

    def has_discrete(self):
        ''' returns True if the model has a discrete
            as it's first (in the case of parallel) reparameterizer'''
        raise NotImplementedError("has_discrete not implemented")

    def get_reparameterizer_scalars(self):
        ''' returns a map of the scalars of the reparameterizers.
            This is useful for visualization purposes'''
        raise NotImplementedError("get_reparameterizer_scalars not implemented")

    def reparameterize(self, logits):
        ''' reparameterizes the latent logits appropriately '''
        raise NotImplementedError("reparameterize not implemented")

    def decode(self, z):
        '''returns logits '''
        raise NotImplementedError("decode not implemented")

    def posterior(self, x):
        ''' get a reparameterized Q(z|x) for a given x '''
        z_logits = self.encode(x)
        return self.reparameterize(z_logits)

    def encode(self, x):
        ''' encodes via a convolution and returns logits '''
        raise NotImplementedError("encode not implemented")

    def generate(self, z):
        ''' returns a generation for a given z '''
        raise NotImplementedError("generate not implemented")

    def kld(self, dist_params):
        ''' KL divergence between dist_a and prior '''
        raise NotImplementedError("kld not implemented")

    def mut_info(self, dist_params):
        ''' helper to get the mutual info to add to the loss '''
        raise NotImplementedError("mut_info not implemented")
