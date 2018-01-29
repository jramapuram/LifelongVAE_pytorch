from __future__ import print_function
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy

from models.gumbel import GumbelSoftmax
from models.vae import VAE
from helpers.utils import expand_dims, long_type, squeeze_expand_dim, \
    ones_like, float_type, pad


def detach_from_graph(param_map):
    for _, v in param_map.items():
        if isinstance(v, dict):
            detach_from_graph(v)
        else:
            v = v.detach_()


# def kl_categorical_categorical(dist_a, dist_b, from_index=0, cuda=False):
#     # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/distributions/python/ops/categorical.py
#     dist_a_log_softmax = F.log_softmax(dist_a['logits'][from_index:], dim=-1)
#     dist_a_softmax = F.softmax(dist_a['logits'][from_index:], dim=-1)
#     dist_b_log_softmax = F.log_softmax(dist_b['logits'][from_index:], dim=-1)

#     # zero pad the smaller categorical
#     dist_a_log_softmax, dist_b_log_softmax \
#         = zero_pad_smaller_cat(dist_a_log_softmax,
#                                dist_b_log_softmax,
#                                cuda=cuda)
#     dist_a_softmax, dist_b_log_softmax \
#         = zero_pad_smaller_cat(dist_a_softmax,
#                                dist_b_log_softmax,
#                                cuda=cuda)
#     delta_log_probs1 = dist_a_log_softmax - dist_b_log_softmax
#     return torch.sum(dist_a_softmax * delta_log_probs1, dim=-1)


def kl_categorical_categorical(dist_a, dist_b, from_index=0, cuda=False):
    # dist_a_logits, dist_b_logits \
    #     = zero_pad_smaller_cat(dist_a['logits'][from_index:],
    #                            dist_b['logits'][from_index:],
    #                            cuda=cuda)
    dist_a_softmax = F.softmax(dist_a['logits'][from_index:], dim=-1)
    dist_b_softmax = F.softmax(dist_b['logits'][from_index:], dim=-1)
    dist_a_softmax, dist_b_softmax \
        = zero_pad_smaller_cat(dist_a_softmax,
                               dist_b_softmax,
                               cuda=cuda)
    return torch.sum(dist_a_softmax * torch.log(dist_a_softmax / (dist_b_softmax + 1e-9)), dim=-1)


def kl_isotropic_gauss_gauss(dist_a, dist_b, eps=1e-9):
    # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/distributions/python/ops/normal.py
    sigma_a_sq = dist_a['logvar'].pow(2)
    sigma_b_sq = dist_b['logvar'].pow(2) + eps
    ratio = sigma_a_sq / sigma_b_sq
    return torch.sum(torch.pow(dist_a['mu'] - dist_b['mu'], 2) / (2 * sigma_b_sq)
                     + 0.5 * (ratio - 1 - torch.log(ratio + eps)), dim=-1)


def zero_pad_smaller_cat(cat1, cat2, cuda=False):
    c1shp = cat1.size()
    c2shp = cat2.size()
    diff = abs(c1shp[1] - c2shp[1])

    # blend in extra zeros appropriately
    if c1shp[1] > c2shp[1]:
        cat2 = torch.cat(
            [cat2,
             Variable(float_type(cuda)(c2shp[0], diff).zero_())],
            dim=-1)
    elif c2shp[1] > c1shp[1]:
        cat1 = torch.cat(
            [cat1,
             Variable(float_type(cuda)(c1shp[0], diff).zero_())],
            dim=-1)

    return [cat1, cat2]


class StudentTeacher(nn.Module):
    def __init__(self, initial_model, **kwargs):
        ''' Helper to keep the student-teacher architecture '''
        super(StudentTeacher, self).__init__()
        self.teacher = None
        self.student = initial_model
        self.current_model = 0
        self.ratio = 1.0
        self.num_teacher_samples = None
        self.num_student_samples = None

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def get_name(self):
        #teacher_name = "teacher_" + self.teacher.get_name() if self.teacher else "teacher_none"
        #return teacher_name + "student_" + self.student.get_name()
        return str(self.current_model) + self.student.get_name()

    def posterior_regularizer(self, q_z_given_x_t, q_z_given_x_s):
        ''' Evaluates KL(Q_{\Phi})(z | \hat{x}) || Q_{\phi})(z | \hat{x})) '''
        if 'discrete' in q_z_given_x_s and 'discrete' in q_z_given_x_t:
            return kl_categorical_categorical(q_z_given_x_s['discrete'],
                                              q_z_given_x_t['discrete'],
                                              from_index=self.num_student_samples,
                                              cuda=self.config['cuda'])
        elif 'gaussian' in q_z_given_x_s and 'gaussian' in q_z_given_x_t:
            # gauss kl-kl doesnt have any from-index
            return kl_isotropic_gauss_gauss(q_z_given_x_s['gaussian'],
                                            q_z_given_x_t['gaussian'])
        else:
            raise NotImplementedError("unknown distribution requested for kl")

    def loss_function(self, output_map):
        ''' returns a combined loss of the VAE loss
            + the combiner loss '''
        vae_loss = self.student.loss_function(output_map['student']['x_reconstr_logits'],
                                              output_map['augmented']['data'],
                                              output_map['student']['params'])
        if 'teacher' in output_map:
            posterior_regularizer = self.posterior_regularizer(output_map['teacher']['params'],
                                                               output_map['student']['params'])
            diff = int(np.abs(vae_loss['loss'].size(0) - posterior_regularizer.size(0)))
            posterior_regularizer = pad(posterior_regularizer,
                                        diff,
                                        dim=-1,
                                        prepend=True,
                                        cuda=self.config['cuda'])
            vae_loss['loss'] = torch.mean(vae_loss['loss'] + posterior_regularizer)
            #vae_loss['loss'] = torch.mean(vae_loss['loss']) + posterior_regularizer
            vae_loss['posterior_regularizer_mean'] = torch.mean(posterior_regularizer)

        return vae_loss

    @staticmethod
    def copy_model(src, dest, disable_dst_grads=False):
        src_params = list(src.parameters())
        dest_params = list(dest.parameters())
        for i in range(len(src_params)):
            if src_params[i].size() == dest_params[i].size():
                dest_params[i].data[:] = src_params[i].data[:].clone()

            if disable_dst_grads:
                dest_params[i].requires_grad = False

        return [src, dest]

    def fork(self):
        config_copy = deepcopy(self.student.config)
        config_copy['discrete_size'] += 1
        self.teacher = deepcopy(self.student)

        # self.teacher = VAE(input_shape=self.student.input_shape,
        #                    **{'kwargs': self.student.config}
        # )
        # data = float_type(self.config['cuda'])(self.student.config['batch_size'],
        #                                        *self.student.input_shape).normal_()
        # self.teacher(Variable(data))
        # self.student, self.teacher \
        #     = self.copy_model(self.student,
        #                       self.teacher,
        #                       disable_dst_grads=True)

        # student_params = list(self.student.parameters())
        # teacher_params = list(self.teacher.parameters())
        # for i in range(len(student_params)):
        #     teacher_params[i].data[:] = student_params[i].data[:].clone()
        #     teacher_params[i].requires_grad = False

        self.student = VAE(input_shape=self.teacher.input_shape,
                           **{'kwargs': config_copy}
        )

        # forward pass once to build lazy modules
        # data = float_type(self.config['cuda'])(self.student.config['batch_size'],
        #                                        *self.student.input_shape).normal_()
        # self.student(Variable(data))

        # # copy teacher params into student while
        # # increasing the dimensionality of the projection
        # self.teacher, self.student \
        #     = self.copy_model(self.teacher, self.student, disable_dst_grads=False)

        # copy the teacher weights into the student model
        # for student_param, teacher_param in zip(self.student.parameters(),
        #                                         self.teacher.parameters()):
        #     student_size = student_param.size()
        #     teacher_size = teacher_param.size()
        #     if student_size == teacher_size:
        #         student_param.data = teacher_param.data.clone()

        # update the current model's ratio
        self.current_model += 1
        self.ratio = self.current_model / (self.current_model + 1.0)

    def generate_synthetic_samples(self, model, batch_size):
        z_samples = model.reparameterizer.prior([batch_size,
                                                 model.reparameterizer.output_size])
        return model.nll_activation(model.decode(z_samples))

    def _augment_data(self, x):
        ''' return batch_size worth of samples that are augmented
            from the teacher model '''
        if self.ratio == 1.0 or not self.training:
            return x   # base case

        batch_size = x.size(0)
        self.num_teacher_samples = int(batch_size * self.ratio)
        self.num_student_samples = max(batch_size - self.num_teacher_samples, 1)
        generated_teacher_samples = self.generate_synthetic_samples(self.teacher, batch_size)
        return torch.cat([x[0:self.num_student_samples],
                          generated_teacher_samples[0:self.num_teacher_samples]], 0)

    def forward(self, x):
        x_augmented = self._augment_data(x)
        x_recon_student, params_student = self.student(x_augmented)
        ret_map = {
            'student':{
                'params': params_student,
                'x_reconstr': self.student.nll_activation(x_recon_student),
                'x_reconstr_logits': x_recon_student
            },
            'augmented': {
                'data': x_augmented,
                'num_student': self.num_student_samples,
                'num_teacher': self.num_teacher_samples
            }
        }

        # encode teacher with synthetic data
        if self.teacher is not None:
            # only teacher Q(z|x) is needed, so dont run decode step
            z_logits_teacher = self.teacher.encode(x_augmented)
            _, params_teacher = self.teacher.reparameterize(z_logits_teacher)
            # detach_from_graph(params_teacher)
            ret_map['teacher']= {
                'params': params_teacher
            }

        return ret_map
