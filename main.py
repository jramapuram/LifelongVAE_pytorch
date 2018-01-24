import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from copy import deepcopy

from models.vae import VAE
from models.student_teacher import StudentTeacher
from datasets.class_sampler import ClassSampler
from datasets.cifar import CIFAR10Loader
from datasets.fashion_mnist import FashionMNISTLoader
from datasets.mnist_cluttered import ClutteredMNISTLoader
from datasets.mnist import MNISTLoader
from datasets.svhn import SVHNCenteredLoader, SVHNFullLoader
from helpers.grapher import Grapher
from helpers.utils import to_data, softmax_accuracy, expand_dims, \
    int_type, float_type, long_type, add_weight_norm, ones_like, \
    squeeze_expand_dim

parser = argparse.ArgumentParser(description='LifeLong VAE Pytorch')

# Task parameters
parser.add_argument('--task', type=str, default="mnist",
                    help="task to work on [mnist / cifar10 / fashion / svhn_centered / svhn / clutter] (default: mnist)")
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='minimum number of epochs to train (default: 10)')
parser.add_argument('--continuous-size', type=int, default=32, metavar='L',
                    help='latent size of continuous variable when using mixture or gaussian (default: 32)')
parser.add_argument('--discrete-size', type=int, default=1,
                    help='initial dim of discrete variable when using mixture or gumbel (default: 1)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--data-dir', type=str, default='./data_dir',
                    metavar='DD',
                    help='directory which contains data')

# Model parameters
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--filter-depth', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--reparam-type', type=str, default='isotropic_gaussian',
                    help='isotropic_gaussian, discrete or mixture [default: isotropic_gaussian]')
parser.add_argument('--layer-type', type=str, default='conv',
                    help='dense or conv (default: conv)')
parser.add_argument('--nll-type', type=str, default='bernoulli',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--mut-reg', type=float, default=0.3,
                    help='mutual information regularizer [mixture only] (default: 0.3)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# Optimizer
parser.add_argument('--optimizer', type=str, default="rmsprop",
                    help="specify optimizer (default: rmsprop)")

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port', type=int, default="8097",
                    help='visdom port for graphs (default: 8097)')

# Device parameters
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# handle randomness / non-randomness
if args.seed is not None:
    print("setting seed %d" % args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed_all(args.seed)


# Global counter
TOTAL_ITER = 0


def build_optimizer(model):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    return optim_map[args.optimizer.lower().strip()](
        model.parameters(), lr=args.lr
    )

def train(epoch, model, optimizer, data_loader, grapher):
    global TOTAL_ITER
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader.train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        if len(list(target.size())) > 1:  #XXX: hax
            target = torch.squeeze(target)

        optimizer.zero_grad()

        # run the VAE + the DNN on the latent space
        output_map = model(data)
        loss = model.loss_function(output_map) # vae loss terms

        # compute loss
        #loss.backward(retain_graph=True)
        loss['loss_mean'].backward()
        optimizer.step()

        # log every nth interval
        if batch_idx % args.log_interval == 0:
            # the total number of samples is different
            # if we have filtered using the class_sampler
            if hasattr(data_loader.train_loader, "sampler") \
               and hasattr(data_loader.train_loader.sampler, "num_samples"):
                num_samples = data_loader.train_loader.sampler.num_samples
            else:
                num_samples = len(data_loader.train_loader)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKLD: {:.4f}\tNLL: {:.4f}'.format(
                epoch, batch_idx * len(data), num_samples,
                100. * batch_idx * len(data) / num_samples,
                loss['loss_mean'].data[0], loss['kld_mean'].data[0], loss['nll_mean'].data[0]))

            grapher.register_single({'train_loss': [[TOTAL_ITER], [loss['loss_mean'].data[0]]]},
                                    plot_type='line')
            grapher.register_single({'train_kld': [[TOTAL_ITER], [loss['kld_mean'].data[0]]]},
                                    plot_type='line')
            grapher.register_single({'train_nll': [[TOTAL_ITER], [loss['nll_mean'].data[0]]]},
                                    plot_type='line')
            grapher.register_single({'train_elbo': [[TOTAL_ITER], [loss['elbo_mean'].data[0]]]},
                                    plot_type='line')
            register_images(output_map['student']['x_reconstr'],
                            output_map['augmented']['data'],
                            grapher)
            grapher.show()


        TOTAL_ITER += 1


def register_images(reconstr_x, data, grapher, prefix="train"):
    reconstr_x = torch.min(reconstr_x, ones_like(reconstr_x, args.cuda))
    vis_x = torch.min(data, ones_like(data, args.cuda))
    grapher.register_single({'%s_reconstructions' % prefix: reconstr_x}, plot_type='imgs')
    grapher.register_single({'%s_inputs' % prefix: vis_x}, plot_type='imgs')


def test(epoch, model, data_loader, grapher):
    model.eval()
    test_loss = []
    test_kld = []
    test_nll = []
    test_elbo = []

    for data, target in data_loader.test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        if len(list(target.size())) > 1:  #XXX: hax
            target = torch.squeeze(target)

        output_map = model(data)
        loss_t = model.loss_function(output_map) # vae loss terms
        test_loss += [loss_t['loss_mean'].data[0]]
        test_kld += [loss_t['kld_mean'].data[0]]
        test_nll += [loss_t['nll_mean'].data[0]]
        test_elbo += [loss_t['elbo_mean'].data[0]]

    test_nll = np.mean(test_nll)
    test_kld = np.mean(test_kld)
    test_loss = np.mean(test_loss)
    test_elbo = np.mean(test_elbo)

    print('\nTest set: Average loss: {:.4f}\tKLD: {:.4f}\tNLL: {:.4f}\n'.format(
        test_loss, test_kld, test_nll))

    # plot the test accuracy and loss
    grapher.register_single({'test_loss': [[epoch], [test_loss]]}, plot_type='line')
    grapher.register_single({'test_kld': [[epoch], [test_kld]]}, plot_type='line')
    grapher.register_single({'test_nll': [[epoch], [test_nll]]}, plot_type='line')
    grapher.register_single({'test_elbo': [[epoch], [test_elbo]]}, plot_type='line')
    register_images(output_map['student']['x_reconstr'],
                    output_map['augmented']['data'],
                    grapher, 'test')
    grapher.show()


def generate(model, grapher):
    model.eval()
    # gen = model.student.reparameterizer.prior(
    #     [args.batch_size, model.student.reparameterizer.output_size]
    # )
    # gen =  model.student.nll_activation(model.student.decode(gen))
    gen = model.generate_synthetic_samples(model.student,
                                           args.batch_size)
    gen = torch.min(gen, ones_like(gen, args.cuda))
    grapher.register_single({'generated': gen}, plot_type='imgs')


def get_samplers(num_classes):
    ''' builds samplers taking into account previous classes'''
    test_samplers = []
    for i in range(num_classes):
        numbers = list(range(i + 1)) if i > 0 else 0
        test_samplers.append(lambda x, j=numbers: ClassSampler(x, class_number=j))

    train_samplers = [lambda x, j=j: ClassSampler(x, class_number=j)
                      for j in range(num_classes)]
    return train_samplers, test_samplers

def get_model_and_loader():
    ''' helper to return the model and the loader '''
    # we build 10 samplers as all of the below have 10 classes
    train_samplers, test_samplers = get_samplers(num_classes=10)

    if args.task == 'cifar10':
        loaders = [CIFAR10Loader(path=args.data_dir,
                                 batch_size=args.batch_size,
                                  train_sampler=tr,
                                  test_sampler=te,
                                 use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif args.task == 'mnist':
        loaders = [MNISTLoader(path=args.data_dir,
                               batch_size=args.batch_size,
                               train_sampler=tr,
                               test_sampler=te,
                               use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif args.task == 'clutter':
        loaders = [ClutteredMNISTLoader(path=args.data_dir,
                                        batch_size=args.batch_size,
                                        train_sampler=tr,
                                        test_sampler=te,
                                        use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]

    elif args.task == 'svhn':
        loaders = [SVHNCenteredLoader(path=args.data_dir,
                                      batch_size=args.batch_size,
                                      train_sampler=tr,
                                      test_sampler=te,
                                      use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif args.task == 'svhn_centered':
        loaders = [SVHNFullLoader(path=args.data_dir,
                                  batch_size=args.batch_size,
                                  train_sampler=tr,
                                  test_sampler=te,
                                  use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    else:
        raise Exception("unknown dataset provided / not supported yet")

    # append the image shape to the config & build the VAE
    args.img_shp =  loaders[0].img_shp,
    vae = VAE(loaders[0].img_shp,
              kwargs=vars(args))

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(vae, kwargs=vars(args))

    # build the grapher object
    grapher = Grapher(env=student_teacher.get_name(),
                      server=args.visdom_url,
                      port=args.visdom_port)

    return [student_teacher, loaders, grapher]


def lazy_generate_modules(model, img_shp):
    ''' Super hax, but needed for building lazy modules '''
    data = float_type(args.cuda)(args.batch_size, *img_shp).normal_()
    model(Variable(data))


def run(args):
    # collect our model and data loader
    model, data_loaders, grapher = get_model_and_loader()

    # since some modules are lazy generated
    # we want to run a single fwd pass
    lazy_generate_modules(model, data_loaders[0].img_shp)

    # collect our optimizer
    optimizer = build_optimizer(model.student)

    # main training loop
    for j, loader in enumerate(data_loaders):
        num_epochs = args.epochs + np.random.randint(0, 13)
        print("training current distribution for {} epochs".format(num_epochs))
        for epoch in range(1, num_epochs + 1):
            train(epoch, model, optimizer, loader, grapher)
            test(epoch, model, loader, grapher)
            generate(model, grapher)

        if j != len(data_loaders) - 1:
            # spawn a new student & rebuild grapher
            # we also pass the new model's parameters through
            # a new optimizer
            model.fork()
            lazy_generate_modules(model, data_loaders[0].img_shp)
            optimizer = build_optimizer(model.student)
            grapher = Grapher(env=model.get_name(),
                              server=args.visdom_url,
                              port=args.visdom_port)


if __name__ == "__main__":
    run(args)
