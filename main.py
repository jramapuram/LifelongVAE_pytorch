import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

from torch.autograd import Variable

from models.fid import train_fid_model
from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE
from models.student_teacher import StudentTeacher
from models.layers import Submodel, EarlyStopping, init_weights
from datasets.loader import get_sequential_data_loaders, get_loader
from optimizers.adamnormgrad import AdamNormGrad
from helpers.grapher import Grapher
from helpers.utils import to_data, float_type, ones_like, \
    frechet_gauss_gauss, frechet_gauss_gauss_np, \
    append_to_csv, num_samples_in_loader, zero_pad_smaller_cat

parser = argparse.ArgumentParser(description='LifeLong VAE Pytorch')

# Task parameters
parser.add_argument('--uid', type=str, default="",
                    help="add a custom task-specific unique id; appended to name (default: None)")
parser.add_argument('--task', nargs='+', default="mnist",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / merged] (default: mnist)""")
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
parser.add_argument('--calculate-fid', action='store_true',
                    help='calculate FID score (default: False)')
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')

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
parser.add_argument('--mut-reg', type=float, default=0.3,
                    help='mutual information regularizer [mixture only] (default: 0.3)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--ewc', action='store_true',
                    help='use the EWC regularizer instead')
parser.add_argument('--vae-type', type=str, default='parallel',
                    help='vae type [sequential or parallel] (default: parallel)')
parser.add_argument('--disable-regularizers', action='store_true',
                    help='disables mutual info and consistency regularizers')
parser.add_argument('--disable-sequential', action='store_true',
                    help='enables standard batch training')
parser.add_argument('--use-relational-encoder', action='store_true',
                    help='uses a relational network as the encoder projection layer')
parser.add_argument('--disable-student-teacher', action='store_true',
                    help='uses a standard VAE without Student-Teacher architecture')

# Optimizer
parser.add_argument('--optimizer', type=str, default="adamnorm",
                    help="specify optimizer (default: rmsprop)")

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


# Global counter
TOTAL_ITER = 0


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


def train(epoch, model, fisher, optimizer, data_loader, grapher):
    global TOTAL_ITER
    model.train()

    for batch_idx, (data, _) in enumerate(data_loader.train_loader):
        data = Variable(data).cuda() if args.cuda else Variable(data)

        # zero gradients on optimizer
        optimizer.zero_grad()

        # run the VAE + the DNN on the latent space
        output_map = model(data)
        loss = model.loss_function(output_map, fisher)

        # compute loss
        loss['loss_mean'].backward()
        optimizer.step()

        # log every nth interval
        if batch_idx % args.log_interval == 0:
            # the total number of samples is different
            # if we have filtered using the class_sampler
            num_samples = num_samples_in_loader(data_loader.train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKLD: {:.4f}\tNLL: {:.4f}'.format(
                epoch, batch_idx * len(data), num_samples,
                100. * batch_idx * len(data) / num_samples,
                loss['loss_mean'].data[0], loss['kld_mean'].data[0], loss['nll_mean'].data[0]))

            # gether scalar values of reparameterizers
            reparam_scalars = model.student.get_reparameterizer_scalars()

            # plot images and lines
            register_plots({**loss, **reparam_scalars}, grapher, epoch=TOTAL_ITER)
            register_images(output_map['student']['x_reconstr'],
                            output_map['augmented']['data'],
                            grapher)
            grapher.show()


        TOTAL_ITER += 1


def register_plots(loss, grapher, epoch, prefix='train'):
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = k.split('_')[0]
            value = v.data[0] if not isinstance(v, (float, np.float32, np.float64)) else v
            grapher.register_single({'%s_%s' % (prefix, key_name): [[epoch], [value]]},
                                    plot_type='line')


def register_images(reconstr_x, data, grapher, prefix="train"):
    reconstr_x = torch.min(reconstr_x, ones_like(reconstr_x))
    vis_x = torch.min(data, ones_like(data))
    grapher.register_single({'%s_reconstructions' % prefix: reconstr_x}, plot_type='imgs')
    grapher.register_single({'%s_inputs' % prefix: vis_x}, plot_type='imgs')


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


def test(epoch, model, data_loader, grapher):
    model.eval()
    loss_map = {}

    for data, _ in data_loader.test_loader:
        data = Variable(data).cuda() if args.cuda else Variable(data)
        with torch.no_grad():
            output_map = model(data)
            loss_t = model.loss_function(output_map) # vae loss terms
            loss_map = _add_loss_map(loss_map, loss_t)

    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    print('\nTest set: Average loss: {:.4f}\tKLD: {:.4f}\tNLL: {:.4f}\n'.format(
        loss_map['loss_mean'].data[0],
        loss_map['kld_mean'].data[0],
        loss_map['nll_mean'].data[0]))

    # gether scalar values of reparameterizers
    reparam_scalars = model.student.get_reparameterizer_scalars()

    # plot the test accuracy and loss
    register_plots({**loss_map, **reparam_scalars}, grapher, epoch=epoch, prefix='test')
    register_images(output_map['student']['x_reconstr'],
                    output_map['augmented']['data'],
                    grapher, 'test')
    grapher.show()

    # return this for early stopping
    loss_val = loss_map['elbo_mean'].detach().data[0]
    loss_map.clear()
    output_map.clear()
    return loss_val


def calculate_fid_between_models(fid_model, model, data_loader,
                                 lifelong_layer_index=-13, fid_layer_index=-4):
    ''' Extract features and computes the FID score for the VAE vs. the classifier
        NOTE: expects a trained fid classifier and model '''
    fid_submodel = Submodel(fid_model, layer_index=fid_layer_index)
    lifelong_submodel = Submodel(model.student, layer_index=lifelong_layer_index)
    fid_submodel.eval()
    lifelong_submodel.eval()

    # keep track of the running frechet dist
    frechet_dist, count = 0.0, 0

    for data, _ in data_loader.test_loader:
        data = Variable(data).cuda() if args.cuda else Variable(data)
        with torch.no_grad():
            batch_size = data.size(0)

            # extract features from both the models
            fid_features = fid_submodel(data).view(batch_size, -1)
            lifelong_features = lifelong_submodel(data).view(batch_size, -1)

            # compute frechet distance
            frechet_dist += frechet_gauss_gauss(
                D.Normal(torch.mean(fid_features, dim=1), torch.var(fid_features, dim=1)),
                D.Normal(torch.mean(lifelong_features, dim=1), torch.var(lifelong_features, dim=1))
            )
            count += 1

    frechet_dist /= count
    frechet_dist = frechet_dist.cpu().numpy()
    print("frechet distance over {} samples: {}\n".format(
        count*args.batch_size, frechet_dist)
    )
    return frechet_dist


def calculate_fid_from_generated_images(fid_model, model, data_loader, fid_layer_index=-4):
    ''' Extract features and computes the FID score for the VAE vs. the classifier
        NOTE: expects a trained fid classifier and model '''
    fid_submodel = Submodel(fid_model, layer_index=fid_layer_index)
    fid_submodel.eval()
    model.eval()

    # calculate how many synthetic images from the student model
    num_test_samples = num_samples_in_loader(data_loader.test_loader)
    num_synthetic = int(np.ceil(num_test_samples // args.batch_size))
    fid, count = 0.0, 0

    with torch.no_grad():
        synthetic = [model.generate_synthetic_samples(model.student, args.batch_size)
                     for _ in range(num_synthetic + 1)]
        for (data, _), generated in zip(data_loader.test_loader, synthetic):
            data = Variable(data).cuda() if args.cuda else Variable(data)
            fid += frechet_gauss_gauss(
                D.Normal(torch.mean(fid_submodel(generated), dim=0), torch.var(fid_submodel(generated), dim=0)),
                D.Normal(torch.mean(fid_submodel(data), dim=0), torch.var(fid_submodel(data), dim=0))
            ).cpu().numpy()
            count += 1

    frechet_dist = fid / count
    print("frechet distance [ {} samples ]: {}\n".format(
        (num_test_samples // args.batch_size) * args.batch_size, frechet_dist)
    )
    return frechet_dist


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
        loaders = get_sequential_data_loaders(args, num_classes=10)

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


def estimate_fisher(model, data_loader, sample_size=1024):
    # modified from github user kuc2477's code
    # sample loglikelihoods from the dataset.
    loglikelihoods = []
    for x, _ in data_loader.train_loader:
        x = Variable(x).cuda() if args.cuda else Variable(x)
        reconstr_x, params = model.teacher(x)
        loss_teacher = model.teacher.loss_function(reconstr_x, x, params)
        loglikelihoods.append(
            loss_teacher['loss']
        )
        if len(loglikelihoods) >= sample_size // args.batch_size:
            break

    # estimate the fisher information of the parameters.
    loglikelihood = torch.cat(loglikelihoods, 0).mean(0)
    loglikelihood_grads = torch.autograd.grad(
        loglikelihood, model.teacher.parameters()
    )
    parameter_names = [
        n.replace('.', '__') for n, p in model.teacher.named_parameters()
    ]
    return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}


def lazy_generate_modules(model, img_shp):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    data = float_type(args.cuda)(args.batch_size, *img_shp).normal_()
    model(Variable(data))

def calculate_consistency(model, loader):
    ''' \sum z_d(teacher) == z_d(student) forall test samples '''
    consistency = 0.0

    if model.current_model > 0 and (args.reparam_type == 'mixture'
                                    or args.reparam_type == 'discrete'):
        model.eval() # prevents data augmentation
        consistency = []

        for img, _ in loader.test_loader:
            with torch.no_grad():
                img = Variable(img).cuda() if args.cuda else Variable(img)

                output_map = model(img)
                teacher_posterior = output_map['teacher']['params']['discrete']['logits']
                student_posterior = output_map['student']['params']['discrete']['logits']
                teacher_posterior = F.softmax(teacher_posterior, dim=-1)
                student_posterior = F.softmax(student_posterior, dim=-1)
                teacher_posterior, student_posterior \
                    = zero_pad_smaller_cat(teacher_posterior, student_posterior,
                                           cuda=args.cuda)

                correct = to_data(teacher_posterior).max(1)[1] \
                          == to_data(student_posterior).max(1)[1]
                consistency.append(torch.mean(correct.type(float_type(args.cuda))))
                # print("teacher = ", teacher_posterior)
                # print("student = ", student_posterior)
                # print("consistency[-1]=", correct)

        num_test_samples = num_samples_in_loader(loader.test_loader)
        consistency = np.mean(consistency)
        print("Consistency [#samples: {}]: {}\n".format(num_test_samples,
                                                      consistency))
    return np.asarray([consistency])


def calculate_fid(fid_model, model, loader, grapher, across_models=False):
    # evaluate and cache away the FID score
    fid = np.inf
    if args.calculate_fid:
        if across_models:
            # weird way
            fid = calculate_fid_between_models(fid_model, model, loader)
        else:
            # standard way
            fid = calculate_fid_from_generated_images(fid_model, model, loader)

        grapher.vis.text(str(fid), opts=dict(title="FID"))

    return fid

def test_and_generate(epoch, model, loader, grapher):
    test_loss = test(epoch, model, loader, grapher)
    generate(model, grapher, 'student') # generate student samples
    generate(model, grapher, 'teacher') # generate teacher samples
    return test_loss

def run(args):
    # collect our model and data loader
    model, data_loaders, grapher = get_model_and_loader()
    fisher = None

    # since some modules are lazy generated
    # we want to run a single fwd pass
    lazy_generate_modules(model, data_loaders[0].img_shp)

    # collect our optimizer
    optimizer = build_optimizer(model.student)
    print("there are {} params in the st-model and {} params in the student".format(
        len(list(model.parameters())), len(list(model.student.parameters()))))

    # build a classifier to use for FID
    if args.calculate_fid:
        fid_model = train_fid_model(model.student.reparameterizer.input_size,
                                    model.student.reparameterizer.output_size,
                                    args)

    # main training loop
    for j, loader in enumerate(data_loaders):
        num_epochs = args.epochs #+ np.random.randint(0, 13)
        print("training current distribution for {} epochs".format(num_epochs))
        early = EarlyStopping(model, max_steps=50) if args.early_stop else None

        test_loss = 0.
        for epoch in range(1, num_epochs + 1):
            train(epoch, model, fisher, optimizer, loader, grapher)
            test_loss = test(epoch, model, loader, grapher)
            if args.early_stop and early(test_loss):
                early.restore() # restore and test+generate again
                test_loss = test_and_generate(epoch, model, loader, grapher)
                break

            generate(model, grapher, 'student') # generate student samples
            generate(model, grapher, 'teacher') # generate teacher samples

        # evaluate one-time metrics
        append_to_csv([test_loss], "{}_test_elbo.csv".format(args.uid))
        append_to_csv(calculate_consistency(model, loader),
                      "{}_consistency.csv".format(args.uid))
        if args.calculate_fid:
            append_to_csv(calculate_fid(fid_model, model, loader, grapher),
                          "{}_fid.csv".format(args.uid))

        if j != len(data_loaders) - 1:
            # spawn a new student & rebuild grapher; we also pass
            # the new model's parameters through a new optimizer.
            if not args.disable_student_teacher:
                model.fork()
                lazy_generate_modules(model, data_loaders[0].img_shp)
                optimizer = build_optimizer(model.student)
                print("there are {} params in the st-model and {} params in the student".format(
                    len(list(model.parameters())), len(list(model.student.parameters()))))

            grapher.save() # save to json after distributional interval
            grapher = Grapher(env=model.get_name(),
                              server=args.visdom_url,
                              port=args.visdom_port)

            if args.ewc:
                # calculate the fisher from the previous data loader
                fisher = estimate_fisher(model, loader) if args.ewc else None


if __name__ == "__main__":
    run(args)
