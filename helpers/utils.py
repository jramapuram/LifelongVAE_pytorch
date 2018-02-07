import os
import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable


def softmax_accuracy(preds, targets, size_average=True):
    pred = to_data(preds).max(1)[1] # get the index of the max log-probability
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(pred.eq(to_data(targets)).cpu().type(torch.FloatTensor))


def bce_accuracy(pred_logits, targets, cuda=False, size_average=True):
    # pred = F.sigmoid(to_data(pred_logits)) >= 0.5
    # pred = pred.type(int_type(cuda))
    # reduction_fn = torch.mean if size_average is True else torch.sum
    # return reduction_fn(pred.data.eq(to_data(targets)).cpu().type(torch.FloatTensor))

    pred = torch.round(F.sigmoid(to_data(pred_logits)))
    pred = pred.type(int_type(cuda))
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(pred.data.eq(to_data(targets)).cpu().type(torch.FloatTensor), -1)


def expand_dims(tensor, dim=0):
    shape = list(tensor.size())
    shape.insert(dim, 1)
    return tensor.view(*shape)


def squeeze_expand_dim(tensor, axis):
    ''' helper to squeeze a multi-dim tensor and then
        unsqueeze the axis dimension if dims < 4'''
    tensor = torch.squeeze(tensor)
    if len(list(tensor.size())) < 4:
        return tensor.unsqueeze(axis)
    else:
        return tensor


def invert_shuffle(arr, perm):
    for _ in range(arr.size(0) - 1):
        arr = arr[perm]

    return arr


def normalize_images(imgs, mu=None, sigma=None, eps=1e-9):
    ''' normalize imgs with provided mu /sigma
        or computes them and returns with the normalized
       images '''
    if mu is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            mu = np.asarray(
                [np.mean(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            mu = np.asarray(
                [np.mean(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    if sigma is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            sigma = np.asarray(
                [np.std(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    return (imgs - mu) / (sigma + eps), [mu, sigma]


def num_samples_in_loader(data_loader):
    ''' simple helper to return the correct number of samples
        if we have used our class-splitter filter '''
    if hasattr(data_loader, "sampler") \
       and hasattr(data_loader.sampler, "num_samples"):
        num_samples = data_loader.sampler.num_samples
    else:
        num_samples = len(data_loader.dataset)

    return num_samples


def normalize_train_test_images(train_imgs, test_imgs, eps=1e-9):
    ''' simple helper to take train and test images
        and normalize the test images by the train mu/sigma '''
    assert len(train_imgs.shape) == len(test_imgs.shape) >= 4

    train_imgs , [mu, sigma] = normalize_images(train_imgs, eps=eps)
    return [train_imgs,
            (test_imgs - mu) / (sigma + eps)]


def append_to_csv(data, filename):
    with open(filename, 'ab') as f:
        np.savetxt(f, data, delimiter=",")


def frechet_gauss_gauss_np(synthetic_features, test_features):
    # calculate the statistics required for frechet distance
    # https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    mu_synthetic = np.mean(synthetic_features, axis=0)
    sigma_synthetic = np.cov(synthetic_features, rowvar=False)
    mu_test = np.mean(test_features, axis=0)
    sigma_test = np.cov(test_features, rowvar=False)

    m = np.square(mu_synthetic - mu_test).sum()
    s = sp.linalg.sqrtm(np.dot(sigma_synthetic, sigma_test))
    dist = m + np.trace(sigma_synthetic + sigma_synthetic - 2*s)
    if np.isnan(dist):
        raise Exception("nan occured in FID calculation.")

    return dist


def frechet_gauss_gauss(dist_a, dist_b):
    ''' d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)). '''
    m = torch.pow(dist_a.loc - dist_b.loc, 2).sum()
    s = torch.sqrt(dist_a.scale * dist_b.scale)
    return torch.mean(m + dist_a.scale + dist_b.scale - 2*s)


def zeros_like(tensor, cuda=False):
    shp = tensor.size()
    is_var = type(tensor) == Variable
    tensor_type = type(to_data(tensor))
    if tensor_type == float_type(cuda):
        zeros = float_type(cuda)(*shp).zero_()
    elif tensor_type == int_type(cuda):
        zeros = int_type(cuda)(*shp).zero_()
    elif tensor_type == long_type(cuda):
        zeros = long_type(cuda)(*shp).zero_()
    else:
        raise Exception("unsuported type passed to zeros: ", tensor_type)

    return zeros if not is_var else Variable(zeros)


def ones_like(tensor, cuda=False):
    shp = tensor.size()
    is_var = type(tensor) == Variable
    tensor_type = type(to_data(tensor))
    if tensor_type == float_type(cuda):
        ones = float_type(cuda)(*shp).zero_().add_(1)
    elif tensor_type == int_type(cuda):
        ones = int_type(cuda)(*shp).zero_().add_(1)
    elif tensor_type == long_type(cuda):
        ones = long_type(cuda)(*shp).zero_().add_(1)
    else:
        raise Exception("unsupported type passed to ones: ", tensor_type)

    return ones if not is_var else Variable(ones)


def scale(val, src, dst):
    """Helper to scale val from src range to dst range
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


def generate_random_categorical(num_targets, batch_size, use_cuda=False):
    ''' Helper to return a categorical of [batch_size, num_targets]
        where one of the num_targets are chosen at random[uniformly]'''
    # indices = scale(torch.rand(batch_size), [0, 1], [0, num_targets])
    indices = long_type(use_cuda)(batch_size, 1).random_(0, to=num_targets)
    return one_hot((batch_size, num_targets), indices, use_cuda=use_cuda)


def merge_masks_into_imgs(imgs, masks_list):
    # (B x C x H x W)
    # masks are always 2d + batch, so expand
    masks_gathered = torch.cat([expand_dims(m, 0) for m in masks_list], 0)
    # masks_gathered = masks_gathered.repeat(1, 1, 3, 1, 1)

    # drop the zeros in the G & B channels and the masks in R
    if masks_gathered.size()[2] < 3:
        zeros = torch.zeros(masks_gathered.size())
        if masks_gathered.is_cuda:
            zeros = zeros.cuda()

        masks_gathered = torch.cat([masks_gathered, zeros, zeros], 2)

    #print("masks gathered = ", masks_gathered.size())

    # add C - channel
    imgs_gathered = expand_dims(imgs, 1) if len(imgs.size()) < 4 else imgs
    if imgs_gathered.size()[1] == 1:
        imgs_gathered = torch.cat([imgs_gathered,
                                   imgs_gathered,
                                   imgs_gathered], 1)

    # tile the images over 0th dimension to make 5d
    # imgs_gathered = imgs_gathered.repeat(masks_gathered.size()[0], 1, 1, 1, 1)
    # super_imposed = imgs_gathered + masks_gathered

    # add all the filters onto the mask
    super_imposed = imgs_gathered
    for mask in masks_gathered:
        super_imposed += mask

    # normalize to one everywhere
    ones = torch.ones(super_imposed.size())
    if masks_gathered.is_cuda:
        ones = ones.cuda()

    super_imposed = torch.min(super_imposed.data, ones)
    return super_imposed


def one_hot_np(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat


def one_hot(num_cols, indices, use_cuda=False):
    """ Creates a matrix of one hot vectors.

        - num_cols: int
        - indices: FloatTensor array
    """
    batch_size = indices.size(0)
    mask = long_type(use_cuda)(batch_size, num_cols).fill_(0)
    ones = 1
    if isinstance(indices, Variable):
        ones = Variable(long_type(use_cuda)(indices.size()).fill_(1))
        mask = Variable(mask, volatile=indices.volatile)

    return mask.scatter_(1, indices, ones)


def to_data(tensor_or_var):
    '''simply returns the data'''
    if type(tensor_or_var) is Variable:
        return tensor_or_var.data
    else:
        return tensor_or_var


def float_type(use_cuda):
    return torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def pad(tensor_or_var, num_pad, value=0, prepend=False, dim=-1, cuda=False):
    if num_pad == 0:
        return tensor_or_var

    pad_val = float_type(cuda)(num_pad).zero_() + value
    if isinstance(tensor_or_var, Variable):
        pad_val = Variable(pad_val)

    if not prepend:
        return torch.cat([tensor_or_var, pad_val], dim=dim)

    return torch.cat([pad_val, tensor_or_var], dim=dim)


def int_type(use_cuda):
    return torch.cuda.IntTensor if use_cuda else torch.IntTensor


def long_type(use_cuda):
    return torch.cuda.LongTensor if use_cuda else torch.LongTensor


def oneplus(x):
    return F.softplus(x, beta=1)


def add_weight_norm(module):
    params = [p[0] for p in module.named_parameters()]
    for param in params:
        if 'weight' in param or 'W_' in param or 'U_' in param:
            print("adding wn to ", param)
            module = torch.nn.utils.weight_norm(
                module, param)

    return module


def str_to_activ(str_activ):
    ''' Helper to return a tf activation given a str'''
    str_activ = str_activ.strip().lower()
    if str_activ == "elu":
        return F.elu
    # elif str_activ == "selu":
    #     return selu
    elif str_activ == "sigmoid":
        return F.sigmoid
    elif str_activ == "tanh":
        return F.tanh
    elif str_activ == "oneplus":
        return oneplus
    else:
        raise Exception("invalid activation provided")


def check_or_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        #print(module)
        if  any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)

    model.apply(lambda module: module.register_backward_hook(check_grad))
