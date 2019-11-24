import os
import time
import json
import argparse
import numpy as np
import pprint
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from functools import partial
from torch.autograd import Variable
from copy import deepcopy

from models.vae.simple_vae import SimpleVAE
from models.student_teacher import StudentTeacher
from helpers.layers import ModelSaver, init_weights, append_save_and_load_fns
from datasets.loader import get_split_data_loaders, get_loader, sequential_test_set_merger
from optimizers.adamnormgrad import AdamNormGrad
from helpers.grapher import Grapher
from helpers.async_fid.client import FIDClient
from helpers.metrics import calculate_consistency, calculate_fid, estimate_fisher
from helpers.utils import float_type, ones_like, \
    append_to_csv, num_samples_in_loader, check_or_create_dir, \
    dummy_context, number_of_parameters, get_name, get_aws_instance_id

parser = argparse.ArgumentParser(description='LifeLong VAE Pytorch')

# Task parameters
parser.add_argument('--uid', type=str, default="",
                    help="add a custom task-specific unique id; appended to name (default: None)")
parser.add_argument('--task', type=str, default="mnist",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='minimum number of epochs to train (default: 500)')
parser.add_argument('--continuous-size', type=int, default=32, metavar='L',
                    help='latent size of continuous variable when using mixture or gaussian (default: 32)')
parser.add_argument('--discrete-size', type=int, default=1,
                    help='initial dim of discrete variable when using mixture or gumbel (default: 1)')
parser.add_argument('--latent-size', type=int, default=512,
                    help='latent size for layers in networks (default: 512)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--output-dir', type=str, default='./results', metavar='OD',
                    help='directory which contains csv results')
parser.add_argument('--model-dir', type=str, default='.models', metavar='MD',
                    help='directory which contains trained models')
parser.add_argument('--fid-model-dir', type=str, default='.models',
                    help='directory which contains trained FID models')
parser.add_argument('--calculate-fid-with', type=str, default=None,
                    help='enables FID calc & uses model conv/inceptionv3  (default: None)')
parser.add_argument('--disable-augmentation', action='store_true',
                    help='disables student-teacher data augmentation')

# train / eval or resume modes
parser.add_argument('--resume-training-with', type=int, default=None,
                    help='tries to load the model from model_dir and resume training [use int] (default: None)')
parser.add_argument('--eval-with', type=int, default=None,
                    help='tries to load the model from model_dir and evaluate the test dataset [use int] (default: None)')
parser.add_argument('--eval-with-loader', type=int, default=None,
                    help='if there are many loaders use ONLY this loader [use int] (default: None)')

# Model parameters
parser.add_argument('--filter-depth', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--reparam-type', type=str, default='isotropic_gaussian',
                    help='isotropic_gaussian, discrete or mixture [default: isotropic_gaussian]')
parser.add_argument('--encoder-layer-type', type=str, default='conv',
                    help='dense or conv (default: conv)')
parser.add_argument('--decoder-layer-type', type=str, default='conv',
                    help='dense or conv (default: conv)')
parser.add_argument('--nll-type', type=str, default='bernoulli',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--vae-type', type=str, default='parallel',
                    help='vae type [sequential or parallel] (default: parallel)')
parser.add_argument('--conv-normalization', type=str, default='groupnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--dense-normalization', type=str, default='batchnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--activation', type=str, default='elu',
                    help='activation function (default: elu)')
parser.add_argument('--standard-batch-training', action='store_true',
                    help='enables standard batch training')
parser.add_argument('--shuffle-minibatches', action='store_true',
                    help='shuffles the student\'s minibatch (default: False)')
parser.add_argument('--disable-gated', action='store_true',
                    help='disables gated convolutional/dense structure (default: False)')
parser.add_argument('--disable-student-teacher', action='store_true',
                    help='uses a standard VAE without Student-Teacher architecture')

# Optimization related
parser.add_argument('--optimizer', type=str, default="adamnorm",
                    help="specify optimizer (default: rmsprop)")
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

# Regularizer related
parser.add_argument('--disable-regularizers', action='store_true',
                    help='disables mutual info and consistency regularizers')
parser.add_argument('--monte-carlo-infogain', action='store_true',
                    help='use the MC version of mutual information gain / false is analytic (default: False)')
parser.add_argument('--continuous-mut-info', type=float, default=0.0,
                    help='-continuous_mut_info * I(z_c; x) is applied (opposite dir of disc)(default: 0.0)')
parser.add_argument('--discrete-mut-info', type=float, default=0.0,
                    help='+discrete_mut_info * I(z_d; x) is applied (default: 0.0)')
parser.add_argument('--kl-beta', type=float, default=1.0,
                    help='hyperparameter to scale KL term in ELBO')
parser.add_argument('--generative-scale-var', type=float, default=1.0,
                    help='scale variance of prior in order to capture outliers')
parser.add_argument('--consistency-gamma', type=float, default=1.0,
                    help='consistency_gamma * KL(Q_student | Q_teacher) (default: 1.0)')
parser.add_argument('--likelihood-gamma', type=float, default=0.0,
                    help='log-likelihood regularizer between teacher and student, 0 is disabled (default: 0.0)')
parser.add_argument('--mut-clamp-strategy', type=str, default="clamp",
                    help='clamp mut info by norm / clamp / none (default: clamp)')
parser.add_argument('--mut-clamp-value', type=float, default=100.0,
                    help='max / min clamp value if above strategy is clamp (default: 100.0)')
parser.add_argument('--ewc-gamma', type=float, default=0,
                    help='any value greater than 0 enables EWC with this hyper-parameter (default: 0)')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom URL for graphs (needs http, eg: http://localhost) (default: None)')
parser.add_argument('--visdom-port', type=int, default=None,
                    help='visdom port for graphs (default: None)')

# Device parameters
parser.add_argument('--debug-step', action='store_true', default=False,
                    help='only does one step of the execute_graph function per call instead of all minibatches')
parser.add_argument('--half', action='store_true', default=False,
                    help='enables half precision training')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# add aws job ID to config if it exists
aws_instance_id = get_aws_instance_id()
if aws_instance_id is not None:
    args.instance_id = aws_instance_id

# handle non-randomness
if args.seed is not None:
    print("setting seed %d" % args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def build_optimizer(model):
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


def register_plots(loss, grapher, epoch, prefix='train'):
    ''' helper to register all plots with *_mean and *_scalar '''
    for k, v in loss.items():
        if isinstance(v, dict):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = '-'.join(k.split('_')[0:-1])
            value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
            grapher.add_scalar('{}_{}'.format(prefix, key_name), value, epoch)


def register_images(output_map, grapher, prefix='train'):
    ''' helper to register all plots with *_img and *_imgs
        NOTE: only registers 1 image to avoid MILLION imgs in visdom,
              consider adding epoch for tensorboardX though
    '''
    for k, v in output_map.items():
        if isinstance(v, dict):
            register_images(output_map[k], grapher, prefix=prefix)

        if 'img' in k or 'imgs' in k:
            key_name = '-'.join(k.split('_')[0:-1])
            img = torchvision.utils.make_grid(v, normalize=True, scale_each=True)
            grapher.add_image('{}_{}'.format(prefix, key_name),
                              img.detach(),
                              global_step=0) # dont use step

def _add_loss_map(loss_tm1, loss_t):
    if not loss_tm1: # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k or 'scalar' in k:
                resultant[k] = v.detach()

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k or 'scalar' in k:
            resultant[k] = loss_tm1[k] + v.detach()

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    for k in loss_map.keys():
        if k == 'count':
            continue

        loss_map[k] /= loss_map['count']

    return loss_map


def train(epoch, model, fisher, optimizer, loader, grapher, prefix='train'):
    ''' train loop helper '''
    return execute_graph(epoch=epoch, model=model, fisher=fisher,
                         data_loader=loader, grapher=grapher,
                         optimizer=optimizer, prefix='train')


def test(epoch, model, fisher, loader, grapher, prefix='test'):
     ''' test loop helper '''
     return execute_graph(epoch, model=model, fisher=fisher,
                          data_loader=loader, grapher=grapher,
                          optimizer=None, prefix='test')


def execute_graph(epoch, model, fisher, data_loader, grapher, optimizer=None, prefix='test'):
    ''' execute the graph; when 'train' is in the name the model runs the optimizer '''
    start_time = time.time()
    model.eval() if not 'train' in prefix else model.train()
    assert optimizer is not None if 'train' in prefix else optimizer is None
    loss_map, params, num_samples = {}, {}, 0
    recon_dict = { 'inputs': [], 'recon': [] }

    for data, _ in data_loader:
        data = Variable(data).cuda() if args.cuda else Variable(data)
        rnd_idx = torch.randperm(len(data))
        data = data[rnd_idx] # XXX, dont do for test?

        if 'train' in prefix:
            # zero gradients on optimizer before forward pass
            optimizer.zero_grad()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            # run the VAE and extract loss
            output_map = model(data)
            loss_t = model.loss_function(output_map, fisher)

        if 'train' in prefix:
            # compute bp and optimize
            loss_t['loss_mean'].backward()

            # add norm of vectorized grads to plot
            grad_list = [torch.norm(x.grad).unsqueeze(0) for x in model.parameters()
                         if x.grad is not None]
            loss_t['grad_norm_mean'] = torch.mean(torch.cat(grad_list, 0))
            optimizer.step()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += data.size(0)

        if 'train' not in prefix:
            recon_dict['inputs'].append(output_map['augmented']['data'].detach().cpu().numpy())
            recon_dict['recon'].append(output_map['student']['x_reconstr'].detach().cpu().numpy())

        if args.debug_step: # for testing purposes
            break

    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    print('{}[Epoch {}][{} samples][{:.2f} sec]: Average loss: {:.4f}\tELBO: {:.4f}\tKLD: {:.4f}\tNLL: {:.4f}\tMut: {:.4f}'.format(
        prefix, epoch, num_samples, time.time() - start_time,
        loss_map['loss_mean'].item(),
        loss_map['elbo_mean'].item(),
        loss_map['kld_mean'].item(),
        loss_map['nll_mean'].item(),
        loss_map['mut_info_mean'].item()))

    # gather scalar values of reparameterizers (if they exist)
    reparam_scalars = model.student.get_reparameterizer_scalars()

    # plot the test accuracy, loss and images
    if grapher: # only if grapher is not None
        register_plots({**loss_map, **reparam_scalars}, grapher, epoch=epoch, prefix=prefix)
        imgs_map = {
            'original_imgs': output_map['augmented']['data'],
            'vae_reconstruction_imgs': output_map['student']['x_reconstr']
        }
        register_images(imgs_map, grapher, prefix=prefix)
        grapher.show()

    if len(recon_dict['inputs']) > 0 and len(recon_dict['recon']) > 0:
        recon_dict['inputs'] = np.vstack(recon_dict['inputs'])
        recon_dict['recon'] = np.vstack(recon_dict['recon'])

    # return this for early stopping
    loss_val = {'loss_mean': loss_map['loss_mean'].detach().item(),
                'elbo_mean': loss_map['elbo_mean'].detach().item(),
                'recon': recon_dict}
    loss_map.clear()
    params.clear()
    return loss_val


def generate(epoch, student_teacher, grapher, name='teacher', post_imgs=True):
    if args.decoder_layer_type == 'pixelcnn' and epoch % 10 != 0:
        # XXX: don't generate every epoch for pixelcnn
        return

    model = {
        'teacher': student_teacher.teacher,
        'student': student_teacher.student
    }

    if model[name] is not None: # handle base case
        model[name].eval()

        # random generation
        gen_map = {
            'samples_%s_imgs' % name: student_teacher.generate_synthetic_samples(
                model[name],
                args.batch_size)
        }

        # sequential generation for discrete and mixture reparameterizations
        if args.reparam_type == 'mixture' or args.reparam_type == 'discrete':
            gen_map['sequential_samples_%s_imgs'%name] = \
                student_teacher.generate_synthetic_sequential_samples(model[name]).detach()

        if post_imgs:
            register_images(gen_map, grapher, prefix="generated")

        return gen_map


def save_train_test_examples(loader):
    test_imgs, train_imgs = [], []
    for l in loaders:
        test_imgs_i, _ = l.test_loader.__iter__().__next__()
        train_imgs_i, _ = l.train_loader.__iter__().__next__()
        test_imgs.append(test_imgs_i)
        train_imgs.append(train_imgs_i)

    test_imgs = torch.cat(test_imgs, 0)
    train_imgs = torch.cat(train_imgs, 0)

    torchvision.utils.save_image(test_imgs, 'test.png', normalize=True)
    torchvision.utils.save_image(train_imgs, 'train.png', normalize=True)

def get_model_and_loader():
    ''' helper to return the model and the loader '''
    #resizer = [torchvision.transforms.Resize(size=(28, 28))]# ,
               # torchvision.transforms.Grayscale()]
    if args.standard_batch_training: # vanilla batch training
        loaders = get_loader(args, sequential_test_set_merger=False, transform=None, **vars(args))
        loaders = sequential_test_set_merger(loaders)
        loaders = [loaders] if not isinstance(loaders, list) else loaders
    else: # classes split
        #loaders = get_split_data_loaders(args, num_classes=10, transformer=resizer, **vars(args))
        #loaders = get_split_data_loaders(args, num_classes=10, transformer=resizer, **vars(args))
        loaders = get_split_data_loaders(args, num_classes=10, **vars(args))

    for l in loaders:
        print("train = ", num_samples_in_loader(l.train_loader),
              " | test = ", num_samples_in_loader(l.test_loader))

    # append the image shape to the config & build the VAE
    args.input_shape =  loaders[0].img_shp
    args_copy = deepcopy(args)
    args_copy.num_current_model = 0 if args.resume_training_with is None else args.resume_training_with
    args_copy.resume_training_with = None
    vae = SimpleVAE(args.input_shape, kwargs=vars(args_copy))
    vae.get_name = partial(get_name, args_copy)

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(vae, kwargs=vars(args_copy))
    #student_teacher = init_weights(student_teacher)

    # build the grapher object
    if args.visdom_url is not None:
        grapher = Grapher('visdom',
                          env=student_teacher.get_name(),
                          server=args.visdom_url,
                          port=args.visdom_port)
    else:
        grapher = Grapher('tensorboard', comment=student_teacher.get_name())

    return [student_teacher, loaders, grapher]


def lazy_generate_modules(model, img_shp):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    #data = float_type(args.cuda)(args.batch_size, *img_shp).normal_()
    data = float_type(next(model.parameters()).is_cuda)(args.batch_size, *img_shp).normal_()
    model(Variable(data))

    # push toe model to cuda
    if args.cuda and not next(model.parameters()).is_cuda:
        model.cuda()


def test_and_generate(epoch, model, fisher, loader, grapher):
    test_loss = test(epoch=epoch, model=model,
                     fisher=fisher, loader=loader.test_loader,
                     grapher=grapher, prefix='test')
    generate(epoch, model, grapher, 'student') # generate student samples
    generate(epoch, model, grapher, 'teacher') # generate teacher samples
    return test_loss


def generate_N(N, epoch, model, grapher, name='student'):
    """ Helper to generate N samples

    :param N: number of samples, > batch_size
    :param epoch: current epoch
    :param model: the nn.Module
    :param grapher: the grapher oject
    :param name: the name of the submodule
    :returns: N sample dictionary
    :rtype: dict

    """
    assert N >= args.batch_size, "need at least batch_size"
    container = {}
    num_batches = int(np.ceil(float(N) / args.batch_size))
    for _ in range(num_batches):
        gen_dict = generate(epoch, model, grapher, name, post_imgs=False)
        if gen_dict is None:
            break

        for k, v in gen_dict.items():
            if k in container:
                container[k] = torch.cat([container[k], v.detach()], 0)
            else:
                container[k] = v.detach()

    for k, v in container.items():
        container[k] = container[k][0:N]
        print(k, ' = ', v.shape)

    return container


def importance_weighted_loop(data_loader, model, args, max_minibatches=3):
    importance_elbo_mean = 0.0
    return importance_elbo_mean

    for idx, (data, _) in enumerate(data_loader):
        data = Variable(data).cuda() if args.cuda else Variable(data)
        with torch.no_grad():
            importance_elbo_mean += model.importance_weighted_elbo(data).item()

        if idx > max_minibatches - 1:
            break

    return importance_elbo_mean / float(idx + 1)


def train_loop(data_loaders, model, fid, grapher, args, model_idx=None):
    ''' simple helper to run the entire train loop; not needed for eval modes'''
    optimizer = build_optimizer(model.student)     # collect our optimizer
    print("there are {} params with {} elems in the st-model and {} params in the student with {} elems".format(
        len(list(model.parameters())), number_of_parameters(model),
        len(list(model.student.parameters())), number_of_parameters(model.student))
    )

    # main training loop
    fisher = None
    model_indices = np.arange(len(data_loaders)) if model_idx is None \
        else np.arange(model_idx, len(data_loaders)+1)

    print("model_idx = ", model_indices, " |full len(data)= ", len(data_loaders),
          " | using only ", len(data_loaders[model_idx:]))

    for j, loader in zip(model_indices, data_loaders[model_idx:]):
        num_epochs = args.epochs # TODO: randomize epochs by something like: + np.random.randint(0, 13)
        print("training current distribution for {} epochs".format(num_epochs))
        args_copy = deepcopy(args)
        args_copy.num_current_model = j
        args_copy.resume_training_with = None
        model.student = append_save_and_load_fns(model.student, optimizer, grapher, args_copy)

        saver = ModelSaver(args, model.student,
                           larger_is_better=False,
                           burn_in_interval=20,#int(num_epochs*0.2),
                           max_early_stop_steps=30)
        restore_dict = saver.restore()
        init_epoch = epoch = restore_dict['epoch']
        print("starting training from epoch {}".format(init_epoch))

        test_loss = None
        for epoch in range(init_epoch, num_epochs + 1):
            train(epoch, model, fisher, optimizer, loader.train_loader, grapher)
            test_loss = test(epoch, model, fisher, loader.test_loader, grapher)

            # do one more test if we are early stopping
            if saver(test_loss['elbo_mean'], **test_loss) or epoch == num_epochs or (args.debug_step and epoch == 5):
                restore_dict = saver.restore()
                with torch.no_grad():
                    model.eval()
                    test_loss = test(epoch, model, fisher, loader.test_loader, grapher)
                    test_loss['importance_elbo_mean'] = importance_weighted_loop(loader.test_loader, model, args)
                    print("importance elbo = ", test_loss['importance_elbo_mean'])
                    grapher.add_text('importance_elbo', str(test_loss['importance_elbo_mean']), 0)

                    pred_dict = {
                        'student': generate_N(10000, epoch, model, grapher, 'student'),
                        'teacher': generate_N(10000, epoch, model, grapher, 'teacher')
                    }
                    saver.save(**{**restore_dict, **pred_dict, **test_loss})

                break

            generate(epoch, model, grapher, 'student') # generate student samples
            generate(epoch, model, grapher, 'teacher') # generate teacher samples
            if epoch == 2: # make sure we do at least 1 test and train pass
                grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(model.student.config),0)


        # evaluate and save away one-time metrics, these include:
        #    1. test elbo
        #    2. FID
        #    3. consistency
        #    4. num synth + num true samples
        #    5. dump config to visdom
        check_or_create_dir(os.path.join(args.output_dir))
        # append_to_csv([test_loss['elbo_mean']], os.path.join(args.output_dir, "{}_test_elbo.csv".format(args.uid)))
        num_synth_samples = np.ceil(epoch * args.batch_size * model.ratio)
        num_true_samples = np.ceil(epoch * (args.batch_size - (args.batch_size * model.ratio)))
        # append_to_csv([num_synth_samples],os.path.join(args.output_dir, "{}_numsynth.csv".format(args.uid)))
        # append_to_csv([num_true_samples], os.path.join(args.output_dir, "{}_numtrue.csv".format(args.uid)))
        # append_to_csv([epoch], os.path.join(args.output_dir, "{}_epochs.csv".format(args.uid)))
        grapher.add_text('num_synth_samples', str(num_synth_samples), 0)
        grapher.add_text('num_true_samples', str(num_true_samples), 0)
        #grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(model.student.config), 0)


        # calc the consistency using the **PREVIOUS** loader
        if j > 0:
            append_to_csv(calculate_consistency(model, data_loaders[j - 1], args.reparam_type, args.vae_type, args.cuda),
                          os.path.join(args.output_dir, "{}_consistency.csv".format(args.uid)))


        if args.calculate_fid_with is not None:
            # TODO: parameterize num fid samples, currently use less for inceptionv3 as it's COSTLY
            num_fid_samples = 4000 if args.calculate_fid_with != 'inceptionv3' else 1000
            append_to_csv(calculate_fid(fid_model=fid_model,
                                        model=model.student,
                                        loader=loader, grapher=grapher,
                                        num_samples=num_fid_samples,
                                        cuda=args.cuda),
                          os.path.join(args.output_dir, "{}_fid.csv".format(args.uid)))

        grapher.save() # save the remote visdom graphs
        if j != len(data_loaders) - 1:
            if args.ewc_gamma > 0:
                # calculate the fisher from the previous data loader
                print("computing fisher info matrix....")
                fisher_tmp = estimate_fisher(model.student, # this is pre-fork
                                             loader, args.batch_size,
                                             cuda=args.cuda)
                if fisher is not None:
                    assert len(fisher) == len(fisher_tmp), "#fisher params != #new fisher params"
                    for (kf, vf), (kft, vft) in zip(fisher.items(), fisher_tmp.items()):
                        fisher[kf] += fisher_tmp[kft]
                else:
                    fisher = fisher_tmp

            # spawn a new student & rebuild grapher; we also pass
            # the new model's parameters through a new optimizer.
            if not args.disable_student_teacher:
                model.fork()
                lazy_generate_modules(model, data_loaders[0].img_shp)
                optimizer = build_optimizer(model.student)
                print("there are {} params with {} elems in the st-model and {} params in the student with {} elems".format(
                    len(list(model.parameters())), number_of_parameters(model),
                    len(list(model.student.parameters())), number_of_parameters(model.student))
                )

            else:
                # increment anyway for vanilla models
                # so that we can have a separate visdom env
                model.current_model += 1

            if args.visdom_url is not None:
                grapher = Grapher('visdom',
                                  env=model.get_name(),
                                  server=args.visdom_url,
                                  port=args.visdom_port)
            else:
                grapher = Grapher('tensorboard', comment=model.get_name())



def _set_model_indices(model, grapher, idx, img_shp, args):
    def _init_vae(img_shp, config):
        return SimpleVAE(args.input_shape, **{'kwargs': config})

    if idx > 0:         # create some clean models to later load in params
        model.current_model = idx
        if not args.disable_augmentation:
            model.ratio = idx / (idx + 1.0)
            model.current_model = idx
            num_teacher_samples = int(args.batch_size * model.ratio)
            num_student_samples = max(args.batch_size - num_teacher_samples, 1)
            print("#teacher_samples: ", num_teacher_samples,
                  " | #student_samples: ", num_student_samples)

            # copy args and reinit clean models for student and teacher
            config_base = args
            config_teacher = deepcopy(config_base)
            config_student = deepcopy(config_base)
            config_student.num_current_model = idx
            config_teacher.num_current_model = idx - 1
            config_student.resume_training_with = None
            config_teacher.resume_training_with = None

            total_student_discrete = args.discrete_size * (idx + 1)
            config_teacher.discrete_size = total_student_discrete - args.discrete_size
            config_student.discrete_size = total_student_discrete

            model.student = _init_vae(model.student.input_shape, vars(deepcopy(config_student)))
            config_student.discrete_size = args.discrete_size # XXX
            model.student.get_name = partial(get_name, config_student)

            print("student = ", model.student.get_name())

            if not args.disable_student_teacher:
                model.teacher = _init_vae(model.student.input_shape, vars(deepcopy(config_teacher)))
                config_teacher.discrete_size = args.discrete_size # XXX
                model.teacher.get_name = partial(get_name, config_teacher)
                print("teacher = ", model.teacher.get_name())

        # re-init grapher
        grapher = Grapher('visdom', env=model.get_name(),
                          server=args.visdom_url,
                          port=args.visdom_port)

        if model.teacher is not None:
            print("loading teacher model...")
            toptimizer = build_optimizer(model.teacher)     # collect our optimizer
            model.teacher = append_save_and_load_fns(model.teacher, toptimizer, grapher, config_teacher)
            lazy_generate_modules(model, img_shp)
            model.teacher.load()


    return model, grapher


def run(args):
    # collect our model and data loader
    model, data_loaders, grapher = get_model_and_loader()

    # since some modules are lazy generated
    # we want to run a single fwd pass
    lazy_generate_modules(model, data_loaders[0].img_shp)

    print(model)

    # build a classifier to use for FID
    fid_model = None
    if args.calculate_fid_with is not None:
        fid_batch_size = args.batch_size if args.calculate_fid_with == 'conv' else 32
        fid_model = train_fid_model(args,
                                    args.calculate_fid_with,
                                    fid_batch_size)

    #train_loop(data_loaders, model, fid_model, grapher, args, model_idx=args.resume_training_with)

    # handle logic on whether to start /resume training or to eval
    if args.eval_with is None and args.resume_training_with is None:              # normal train loop
        print("starting main training loop from scratch...")
        train_loop(data_loaders, model, fid_model, grapher, args)
    elif args.eval_with is None and args.resume_training_with is not None:    # resume training from latest model
        print("resuming training on model {}...".format(args.resume_training_with))
        model, grapher = _set_model_indices(model, grapher, args.resume_training_with, data_loaders[0].img_shp, args)
        lazy_generate_modules(model, data_loaders[0].img_shp)
        # if not model.load(): # restore after setting model ind
        #     raise Exception("model failed to load for resume training...")

        #train_loop(data_loaders[args.resume_training_with:], model, fid_model, grapher, args,
        train_loop(data_loaders, model, fid_model, grapher, args,
                   model_idx=args.resume_training_with)
    elif args.eval_with is not None:                                      # eval the provided model
        print("evaluating model {}...".format(args.eval_with))
        model, grapher = _set_model_indices(model, grapher, args.eval_with, args)
        lazy_generate_modules(model, data_loaders[0].img_shp)
        if not model.load(): # restore after setting model ind
            raise Exception("model failed to load for resume training...")

        if args.eval_with_loader is not None: # only use 1 loader
            eval_model([data_loaders[args.eval_with_loader]], model, fid_model, args)
        else:
            eval_model(data_loaders, model, fid_model, args)
    else:
        raise Exception("unknown train-eval-resume combo specified!")


if __name__ == "__main__":
    print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))
    run(args)
