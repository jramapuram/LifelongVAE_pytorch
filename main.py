import os
import json
import argparse
import numpy as np
import pprint
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from copy import deepcopy

from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE
from models.student_teacher import StudentTeacher
from helpers.layers import EarlyStopping, init_weights
from datasets.loader import get_split_data_loaders, get_loader
from optimizers.adamnormgrad import AdamNormGrad
from helpers.grapher import Grapher
from helpers.fid import train_fid_model
from helpers.metrics import calculate_consistency, calculate_fid, estimate_fisher
from helpers.utils import float_type, ones_like, \
    append_to_csv, num_samples_in_loader, check_or_create_dir, \
    dummy_context, number_of_parameters

parser = argparse.ArgumentParser(description='LifeLong VAE Pytorch')

# Task parameters
parser.add_argument('--uid', type=str, default="",
                    help="add a custom task-specific unique id; appended to name (default: None)")
parser.add_argument('--task', type=str, default="mnist",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='minimum number of epochs to train (default: 10)')
parser.add_argument('--continuous-size', type=int, default=32, metavar='L',
                    help='latent size of continuous variable when using mixture or gaussian (default: 32)')
parser.add_argument('--discrete-size', type=int, default=1,
                    help='initial dim of discrete variable when using mixture or gumbel (default: 1)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--output-dir', type=str, default='./experiments', metavar='OD',
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
parser.add_argument('--layer-type', type=str, default='conv',
                    help='dense or conv (default: conv)')
parser.add_argument('--nll-type', type=str, default='bernoulli',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--vae-type', type=str, default='parallel',
                    help='vae type [sequential or parallel] (default: parallel)')
parser.add_argument('--normalization', type=str, default='groupnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--activation', type=str, default='elu',
                    help='activation function (default: elu)')
parser.add_argument('--disable-sequential', action='store_true',
                    help='enables standard batch training')
parser.add_argument('--shuffle-minibatches', action='store_true',
                    help='shuffles the student\'s minibatch (default: False)')
parser.add_argument('--use-relational-encoder', action='store_true',
                    help='uses a relational network as the encoder projection layer')
parser.add_argument('--use-pixel-cnn-decoder', action='store_true',
                    help='uses a pixel CNN decoder (default: False)')
parser.add_argument('--disable-gated-conv', action='store_true',
                    help='disables gated convolutional structure (default: False)')
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
parser.add_argument('--kl-reg', type=float, default=1.0,
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
parser.add_argument('--visdom-url', type=str, default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port', type=int, default="8097",
                    help='visdom port for graphs (default: 8097)')

# Device parameters
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
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
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = k.split('_')[0]
            value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
            grapher.register_single({'%s_%s' % (prefix, key_name): [[epoch], [value]]},
                                    plot_type='line')


def register_images(images, names, grapher, prefix="train"):
    ''' helper to register a list of images '''
    if isinstance(images, list):
        assert len(images) == len(names)
        for im, name in zip(images, names):
            register_images(im, name, grapher, prefix=prefix)
    else:
        images = torch.min(images.detach(), ones_like(images))
        grapher.register_single({'{}_{}'.format(prefix, names): images},
                                plot_type='imgs')


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
    model.eval() if not 'train' in prefix else model.train()
    assert optimizer is not None if 'train' in prefix else optimizer is None
    loss_map, params, num_samples = {}, {}, 0

    for data, _ in data_loader:
        data = Variable(data).cuda() if args.cuda else Variable(data)

        if 'train' in prefix:
            # zero gradients on optimizer
            # before forward pass
            optimizer.zero_grad()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            # run the VAE and extract loss
            output_map = model(data)
            loss_t = model.loss_function(output_map, fisher)

        if 'train' in prefix:
            # compute bp and optimize
            loss_t['loss_mean'].backward()
            loss_t['grad_norm_mean'] = torch.norm( # add norm of vectorized grads to plot
                nn.utils.parameters_to_vector(model.parameters())
            )
            optimizer.step()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += data.size(0)

    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    print('{}[Epoch {}][{} samples]: Average loss: {:.4f}\tELBO: {:.4f}\tKLD: {:.4f}\tNLL: {:.4f}\tMut: {:.4f}'.format(
        prefix, epoch, num_samples,
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
        images = [output_map['augmented']['data'], output_map['student']['x_reconstr']]
        img_names = ['original_imgs', 'vae_reconstructions']
        register_images(images, img_names, grapher, prefix=prefix)
        grapher.show()

    # return this for early stopping
    loss_val = {'loss_mean': loss_map['loss_mean'].detach().item(),
                'elbo_mean': loss_map['elbo_mean'].detach().item()}
    loss_map.clear()
    params.clear()
    return loss_val


def generate(student_teacher, grapher, name='teacher'):
    model = {
        'teacher': student_teacher.teacher,
        'student': student_teacher.student
    }

    if model[name] is not None: # handle base case
        model[name].eval()
        # random generation
        gen = student_teacher.generate_synthetic_samples(model[name],
                                                         args.batch_size)
        gen = torch.min(gen, ones_like(gen))
        grapher.register_single({'generated_%s'%name: gen}, plot_type='imgs')

        # sequential generation for discrete and mixture reparameterizations
        if args.reparam_type == 'mixture' or args.reparam_type == 'discrete':
            gen = student_teacher.generate_synthetic_sequential_samples(model[name]).detach()
            gen = torch.min(gen, ones_like(gen))
            grapher.register_single({'sequential_generated_%s'%name: gen}, plot_type='imgs')


def get_model_and_loader():
    ''' helper to return the model and the loader '''
    if args.disable_sequential: # vanilla batch training
        loaders = get_loader(args)
        loaders = [loaders] if not isinstance(loaders, list) else loaders
    else: # classes split
        loaders = get_split_data_loaders(args, num_classes=10)

    for l in loaders:
        print("train = ", num_samples_in_loader(l.train_loader),
              " | test = ", num_samples_in_loader(l.test_loader))

    # append the image shape to the config & build the VAE
    args.img_shp =  loaders[0].img_shp,
    if args.vae_type == 'sequential':
        # Sequential : P(y|x) --> P(z|y, x) --> P(x|z)
        # Keep a separate VAE spawn here in case we want
        # to parameterize the sequence of reparameterizers
        vae = SequentiallyReparameterizedVAE(loaders[0].img_shp,
                                             kwargs=vars(args))
    elif args.vae_type == 'parallel':
        # Ours: [P(y|x), P(z|x)] --> P(x | z)
        vae = ParallellyReparameterizedVAE(loaders[0].img_shp,
                                           kwargs=vars(args))
    else:
        raise Exception("unknown VAE type requested")

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(vae, kwargs=vars(args))
    #student_teacher = init_weights(student_teacher)

    # build the grapher object
    grapher = Grapher(env=student_teacher.get_name(),
                      server=args.visdom_url,
                      port=args.visdom_port)

    return [student_teacher, loaders, grapher]


def lazy_generate_modules(model, img_shp):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    data = float_type(args.cuda)(args.batch_size, *img_shp).normal_()
    model(Variable(data))


def test_and_generate(epoch, model, fisher, loader, grapher):
    test_loss = test(epoch=epoch, model=model,
                     fisher=fisher, loader=loader.test_loader,
                     grapher=grapher, prefix='test')
    generate(model, grapher, 'student') # generate student samples
    generate(model, grapher, 'teacher') # generate teacher samples
    return test_loss


def eval_model(data_loaders, model, fid_model, args):
    ''' simple helper to evaluate the model over all the loaders'''
    for loader in data_loaders:
        test_loss = test(epoch=-1, model=model, fisher=None,
                         loader=loader.test_loader, grapher=None, prefix='test')

        # evaluate and save away one-time metrics
        check_or_create_dir(os.path.join(args.output_dir))
        append_to_csv([test_loss['elbo_mean']], os.path.join(args.output_dir, "{}_test_elbo.csv".format(args.uid)))
        append_to_csv(calculate_consistency(model, loader, args.reparam_type, args.vae_type, args.cuda),
                      os.path.join(args.output_dir, "{}_consistency.csv".format(args.uid)))
        with open(os.path.join(args.output_dir, "{}_conf.json".format(args.uid)), 'w') as f:
            json.dump(model.student.config, f)

        if args.calculate_fid_with is not None:
            # TODO: parameterize num fid samples, currently use less for inceptionv3 as it's COSTLY
            num_fid_samples = 4000 if args.calculate_fid_with != 'inceptionv3' else 1000
            append_to_csv(calculate_fid(fid_model=fid_model,
                                        model=model,
                                        loader=loader, grapher=None,
                                        num_samples=num_fid_samples,
                                        cuda=args.cuda),
                          os.path.join(args.output_dir, "{}_fid.csv".format(args.uid)))


def train_loop(data_loaders, model, fid_model, grapher, args):
    ''' simple helper to run the entire train loop; not needed for eval modes'''
    optimizer = build_optimizer(model.student)     # collect our optimizer
    print("there are {} params with {} elems in the st-model and {} params in the student with {} elems".format(
        len(list(model.parameters())), number_of_parameters(model),
        len(list(model.student.parameters())), number_of_parameters(model.student))
    )

    # main training loop
    fisher = None
    for j, loader in enumerate(data_loaders):
        num_epochs = args.epochs # TODO: randomize epochs by something like: + np.random.randint(0, 13)
        print("training current distribution for {} epochs".format(num_epochs))
        early = EarlyStopping(model, max_steps=50, burn_in_interval=None) if args.early_stop else None
                              #burn_in_interval=int(num_epochs*0.2)) if args.early_stop else None

        test_loss = None
        for epoch in range(1, num_epochs + 1):
            train(epoch, model, fisher, optimizer, loader.train_loader, grapher)
            test_loss = test(epoch, model, fisher, loader.test_loader, grapher)
            if args.early_stop and early(test_loss['loss_mean']):
                early.restore() # restore and test+generate again
                test_loss = test_and_generate(epoch, model, fisher, loader, grapher)
                break

            generate(model, grapher, 'student') # generate student samples
            generate(model, grapher, 'teacher') # generate teacher samples

        # evaluate and save away one-time metrics, these include:
        #    1. test elbo
        #    2. FID
        #    3. consistency
        #    4. num synth + num true samples
        #    5. dump config to visdom
        check_or_create_dir(os.path.join(args.output_dir))
        append_to_csv([test_loss['elbo_mean']], os.path.join(args.output_dir, "{}_test_elbo.csv".format(args.uid)))
        append_to_csv([test_loss['elbo_mean']], os.path.join(args.output_dir, "{}_test_elbo.csv".format(args.uid)))
        num_synth_samples = np.ceil(epoch * args.batch_size * model.ratio)
        num_true_samples = np.ceil(epoch * (args.batch_size - (args.batch_size * model.ratio)))
        append_to_csv([num_synth_samples],os.path.join(args.output_dir, "{}_numsynth.csv".format(args.uid)))
        append_to_csv([num_true_samples], os.path.join(args.output_dir, "{}_numtrue.csv".format(args.uid)))
        append_to_csv([epoch], os.path.join(args.output_dir, "{}_epochs.csv".format(args.uid)))
        grapher.vis.text(num_synth_samples, opts=dict(title="num_synthetic_samples"))
        grapher.vis.text(num_true_samples, opts=dict(title="num_true_samples"))
        grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(model.student.config),
                         opts=dict(title="config"))

        # calc the consistency using the **PREVIOUS** loader
        if j > 0:
            append_to_csv(calculate_consistency(model, data_loaders[j - 1], args.reparam_type, args.vae_type, args.cuda),
                          os.path.join(args.output_dir, "{}_consistency.csv".format(args.uid)))


        if args.calculate_fid_with is not None:
            # TODO: parameterize num fid samples, currently use less for inceptionv3 as it's COSTLY
            num_fid_samples = 4000 if args.calculate_fid_with != 'inceptionv3' else 1000
            append_to_csv(calculate_fid(fid_model=fid_model,
                                        model=model,
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

            grapher = Grapher(env=model.get_name(),
                              server=args.visdom_url,
                              port=args.visdom_port)


def _set_model_indices(model, grapher, idx, args):
    def _init_vae(img_shp, config):
        if args.vae_type == 'sequential':
            # Sequential : P(y|x) --> P(z|y, x) --> P(x|z)
            # Keep a separate VAE spawn here in case we want
            # to parameterize the sequence of reparameterizers
            vae = SequentiallyReparameterizedVAE(img_shp,
                                                 **{'kwargs': config})
        elif args.vae_type == 'parallel':
            # Ours: [P(y|x), P(z|x)] --> P(x | z)
            vae = ParallellyReparameterizedVAE(img_shp,
                                               **{'kwargs': config})
        else:
            raise Exception("unknown VAE type requested")

        return vae

    if idx > 0:         # create some clean models to later load in params
        model.current_model = idx
        if not args.disable_augmentation:
            model.ratio = idx / (idx + 1.0)
            num_teacher_samples = int(args.batch_size * model.ratio)
            num_student_samples = max(args.batch_size - num_teacher_samples, 1)
            print("#teacher_samples: ", num_teacher_samples,
                  " | #student_samples: ", num_student_samples)

            # copy args and reinit clean models for student and teacher
            config_base = vars(args)
            config_teacher = deepcopy(config_base)
            config_student = deepcopy(config_base)
            config_teacher['discrete_size'] += idx - 1
            config_student['discrete_size'] += idx
            model.student = _init_vae(model.student.input_shape, config_student)
            if not args.disable_student_teacher:
                model.teacher = _init_vae(model.student.input_shape, config_teacher)

        # re-init grapher
        grapher = Grapher(env=model.get_name(),
                          server=args.visdom_url,
                          port=args.visdom_port)

    return model, grapher


def run(args):
    # collect our model and data loader
    model, data_loaders, grapher = get_model_and_loader()

    # since some modules are lazy generated
    # we want to run a single fwd pass
    lazy_generate_modules(model, data_loaders[0].img_shp)

    # build a classifier to use for FID
    fid_model = None
    if args.calculate_fid_with is not None:
        fid_batch_size = args.batch_size if args.calculate_fid_with == 'conv' else 32
        fid_model = train_fid_model(args,
                                    args.calculate_fid_with,
                                    fid_batch_size)

    # handle logic on whether to start /resume training or to eval
    if args.eval_with is None and args.resume_training_with is None:              # normal train loop
        print("starting main training loop from scratch...")
        train_loop(data_loaders, model, fid_model, grapher, args)
    elif args.eval_with is None and args.resume_training_with is not None:    # resume training from latest model
        print("resuming training on model {}...".format(args.resume_training_with))
        model, grapher = _set_model_indices(model, grapher, args.resume_training_with, args)
        lazy_generate_modules(model, data_loaders[0].img_shp)
        if not model.load(): # restore after setting model ind
            raise Exception("model failed to load for resume training...")

        train_loop(data_loaders[args.resume_training_with:], model, fid_model, grapher, args)
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
    run(args)
