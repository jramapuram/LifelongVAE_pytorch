from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict

from optimizers.adamnormgrad import AdamNormGrad
from models.layers import View, Identity, flatten_layers, EarlyStopping
from datasets.loader import get_loader
from datasets.utils import GenericLoader, simple_merger
from helpers.utils import float_type, zeros_like, ones_like, \
    softmax_accuracy, check_or_create_dir


def build_optimizer(model, args):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adamnorm": AdamNormGrad,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    return optim_map[args.optimizer.lower().strip()](
        model.parameters(), lr=args.lr
    )


def train(epoch, model, optimizer, data_loader, args):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader.train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        if len(list(target.size())) > 1:  #XXX: hax
            target = torch.squeeze(target)

        optimizer.zero_grad()

        # project to the output dimension
        output = model(data)
        loss = model.loss_function(output, target)
        correct = softmax_accuracy(output, target)

        # compute loss
        loss.backward()
        optimizer.step()

        # log every nth interval
        if batch_idx % args.log_interval == 0:
            # the total number of samples is different
            # if we have filtered using the class_sampler
            if hasattr(data_loader.train_loader, "sampler") \
               and hasattr(data_loader.train_loader.sampler, "num_samples"):
                num_samples = data_loader.train_loader.sampler.num_samples
            else:
                num_samples = len(data_loader.train_loader.dataset)

            print('[FID]Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                epoch, batch_idx * len(data), num_samples,
                100. * batch_idx * len(data) / num_samples,
                loss.data[0], correct))


def test(epoch, model, data_loader, args):
    model.eval()
    loss = []
    correct = []

    for data, target in data_loader.test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            if len(list(target.size())) > 1:  #XXX: hax
                target = torch.squeeze(target)

            output = model(data)
            loss_t = model.loss_function(output, target)
            correct_t = softmax_accuracy(output, target)

            loss.append(loss_t.detach().cpu().data[0])
            correct.append(correct_t)

    loss = np.mean(loss)
    acc = np.mean(correct)
    print('\n[FID]Test Epoch: {}\tAverage loss: {:.4f}\tAverage Accuracy: {:.4f}\n'.format(
        epoch, loss, acc)
    )
    return loss, acc


def lazy_generate_modules(model, img_shp, batch_size, cuda):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    data = float_type(cuda)(batch_size, *img_shp).normal_()
    model(Variable(data))


def train_fid_model(reparameterizer_input_size,
                    reparameterizer_output_size,
                    args):
    ''' builds and trains a classifier '''
    loader = get_loader(args)
    if isinstance(loader, list): # has a sequential loader
        # loader = simple_merger(loader, args.batch_size, args.cuda)
        loader = loader[0]

    model = FID(loader.img_shp,
                loader.output_size,
                reparameterizer_input_size=32,
                reparameterizer_output_size=32,
                kwargs=vars(args))
    if not model.model_exists:
        lazy_generate_modules(model, loader.img_shp,
                              args.batch_size, args.cuda)
        optimizer = build_optimizer(model, args)
        early_stop = EarlyStopping(model, max_steps=10)

        for epoch in range(1, args.epochs + 1):
            train(epoch, model, optimizer, loader, args)
            loss, _ = test(epoch, model, loader, args)
            if early_stop(loss):
                early_stop.restore()
                break

        # save the model
        model.save()

    return model


class FID(nn.Module):
    def __init__(self, input_shape, output_size,
                 reparameterizer_input_size,
                 reparameterizer_output_size,
                 activation_fn=nn.ELU, **kwargs):
        super(FID, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1
        self.activation_fn = activation_fn

        # keep the same structure as the VAE
        self.reparameterizer_input_size = reparameterizer_input_size
        self.reparameterizer_output_size = reparameterizer_output_size

        # grab the meta config
        self.config = kwargs['kwargs']

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.full_model = None
        self.model_exists = self.load()

    def load(self):
        # load the FID model if it exists
        if os.path.isdir(".models"):
            model_filename = os.path.join(".models", self.get_name() + ".th")
            if os.path.isfile(model_filename):
                print("loading existing FID model")
                lazy_generate_modules(self, self.input_shape,
                                      self.config['batch_size'],
                                      self.config['cuda'])
                self.load_state_dict(torch.load(model_filename))
                return True

        return False

    def save(self, overwrite=False):
        # save the FID model if it doesnt exist
        check_or_create_dir(".models")
        model_filename = os.path.join(".models", self.get_name() + ".th")
        if not os.path.isfile(model_filename) or overwrite:
            print("saving existing FID model")
            torch.save(self.state_dict(), model_filename)

    def get_name(self):
        full_hash_str = "_" + str(self.config['layer_type']) + \
                        "_input" + str(self.input_shape) + \
                        "_batch" + str(self.config['batch_size']) + \
                        "_filter_depth" + str(self.config['filter_depth']) + \
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
        # tasks_cleaned = [t.split('_')[1] if 'rotated' in t else t for t in self.config['task']]
        # return 'fid_' + '_'.join(tasks_cleaned) + full_hash_str
        return 'fid_' + '_'.join(self.config['task']) + full_hash_str


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
                nn.Conv2d(self.config['filter_depth']*16, self.reparameterizer_input_size, 1, stride=1, bias=True),
                nn.BatchNorm2d(self.reparameterizer_input_size),
                self.activation_fn(inplace=True)
                # output dim: opt.z_dim x 1 x 1
            )
        elif self.config['layer_type'] == 'dense':
            encoder = nn.Sequential(
                View([-1, int(np.prod(self.input_shape))]),
                nn.Linear(int(np.prod(self.input_shape)), self.reparameterizer_input_size),
                nn.BatchNorm1d(self.reparameterizer_input_size),
                self.activation_fn(),
                nn.Linear(self.reparameterizer_input_size, self.reparameterizer_input_size),
                nn.BatchNorm1d(self.reparameterizer_input_size),
                self.activation_fn(),
                nn.Linear(self.reparameterizer_input_size, self.reparameterizer_input_size),
                nn.BatchNorm1d(self.reparameterizer_input_size),
                self.activation_fn(),
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
        if self.config['layer_type'] == 'conv':
            decoder = nn.Sequential(
                View([-1, self.reparameterizer_output_size, 1, 1]),
                # input dim: z_dim x 1 x 1
                nn.ConvTranspose2d(self.reparameterizer_output_size, self.config['filter_depth']*8, 4, stride=1, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*8),
                self.activation_fn(inplace=True),
                # state dim:   256 x 4 x 4
                nn.ConvTranspose2d(self.config['filter_depth']*8, self.config['filter_depth']*4, 4, stride=2, bias=True),
                nn.BatchNorm2d(self.config['filter_depth']*4),
                self.activation_fn(inplace=True)
            )
        elif self.config['layer_type'] == 'dense':
            decoder = nn.Sequential(
                View([-1, self.reparameterizer_output_size]),
                nn.Linear(self.reparameterizer_output_size, self.reparameterizer_output_size),
                nn.BatchNorm1d(self.reparameterizer_output_size),
                self.activation_fn(),
                nn.Linear(self.reparameterizer_output_size, self.reparameterizer_output_size),
                nn.BatchNorm1d(self.reparameterizer_output_size),
                self.activation_fn(),
                nn.Linear(self.reparameterizer_output_size, int(np.prod(self.input_shape))),
                nn.BatchNorm1d(int(np.prod(self.input_shape))),
                self.activation_fn(),
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

    def encode(self, x):
        z = self.encoder(x)
        conv_output_shp = int(np.prod(z.size()[1:]))

        # project via linear layer
        self._lazy_init_dense(conv_output_shp,
                              self.reparameterizer_output_size,
                              name='fid_enc_proj')
        return self.fid_enc_proj(z)

    def decode(self, z):
        decode_logits =  self.decoder(z)

        # project via linear layer
        conv_output_shp = int(np.prod(decode_logits.size()[1:]))
        self._lazy_init_dense(conv_output_shp,
                              self.output_size,
                              name='fid_dec_proj')
        return self.fid_dec_proj(decode_logits)

    def compile_full_model(self):
        if hasattr(self, 'fid_enc_proj') and hasattr(self, 'fid_dec_proj'):
            if not self.full_model:
                full_model_list, _ = flatten_layers(
                    nn.Sequential(
                        self.encoder,
                        self.fid_enc_proj,
                        self.decoder,
                        self.fid_dec_proj
                    ))
                self.full_model = nn.Sequential(OrderedDict(full_model_list))
        else:
            raise Exception("[FID] cant compile full model till you lazy-init the dense layer")

    def nll_activation(self, logits):
        ''' NOTE: nll is just classification here '''
        return F.log_softmax(logits)

    def loss_function(self, pred, target):
        return F.cross_entropy(pred, target)

    def forward(self, x):
        z_logits = self.encode(x)
        return self.decode(z_logits)
