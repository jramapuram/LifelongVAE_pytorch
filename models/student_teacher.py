from __future__ import print_function
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy

from models.gumbel import GumbelSoftmax
from models.vae import VAE
from helpers.utils import expand_dims, long_type, squeeze_expand_dim, ones_like


def detach_from_graph(param_map):
    for _, v in param_map.items():
        if isinstance(v, dict):
            detach_from_graph(v)
        else:
            v = v.detach_()


def kl_categorical_categorical(dist_a, dist_b, eps=1e-9):
    # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/distributions/python/ops/categorical.py
    delta_log_probs1 = F.log_softmax(dist_a['logits'], dim=-1) \
                       - F.log_softmax(dist_b['logits'], dim=-1)
    return torch.sum(F.softmax(dist_a['logits'], dim=-1) * delta_log_probs1, dim=-1)


def kl_isotropic_gauss_gauss(dist_a, dist_b, eps=1e-9):
    # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/distributions/python/ops/normal.py
    sigma_a_sq = dist_a['logvar'].pow(2)
    sigma_b_sq = dist_b['logvar'].pow(2)
    ratio = sigma_a_sq / sigma_b_sq
    return torch.pow(dist_a['mu'] - dist_b['mu'], 2) / (2 * sigma_b_sq) \
        + 0.5 * (ratio - 1 - torch.log(ratio + eps))


class StudentTeacher(nn.Module):
    def __init__(self, initial_model, **kwargs):
        ''' Helper to keep the student-teacher architecture '''
        super(StudentTeacher, self).__init__()
        self.teacher = None
        self.student = initial_model
        self.current_model = 0
        self.ratio = 1.0

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def get_name(self):
        teacher_name = "teacher_" + self.teacher.get_name() if self.teacher else "teacher_none"
        return teacher_name + "student_" + self.student.get_name()

    def posterior_regularizer(self, q_z_given_x_t, q_z_given_x_s):
        if 'discrete' in q_z_given_x_s and 'discrete' in q_z_given_x_t:
            return torch.mean(kl_categorical_categorical(q_z_given_x_s['discrete'],
                                                         q_z_given_x_t['discrete']))
        elif 'gaussian' in q_z_given_x_s and 'gaussian' in q_z_given_x_t:
            return kl_isotropic_gauss_gauss(q_z_given_x_s['gaussian'],
                                            q_z_given_x_t['gaussian'])
        else:
            raise NotImplementedError("unknown distribution requested for kl")

    def loss_function(self, output_map, x):
        ''' returns a combined loss of the VAE loss
            + the combiner loss '''
        vae_loss = self.student.loss_function(output_map['student']['x_reconstr'], x,
                                              output_map['student']['params'])
        if 'teacher' in output_map:
            posterior_regularizer = self.posterior_regularizer(output_map['teacher']['params'],
                                                               output_map['student']['params'])
            vae_loss['loss'] += posterior_regularizer
            vae_loss['posterior_regularizer'] = posterior_regularizer

        return vae_loss

    def fork(self):
        config_copy = deepcopy(self.config)
        config_copy['mixture_discrete_size'] += 1
        self.teacher = deepcopy(self.student)
        self.student = VAE(input_shape=self.teacher.input_shape,
                           latent_size=self.teacher.latent_size,
                           **{'kwargs': config_copy}
        )

        # copy the teacher weights into the student model
        for student_param, teacher_param in zip(self.student.parameters(),
                                                self.teacher.parameters()):
            student_size = student_param.size()
            teacher_size = teacher_param.size()
            if student_size == teacher_size:
                student_param.data = teacher_param.data.clone()

        # update the current model's ratio
        self.current_model += 1
        self.ratio = self.current_model / (self.current_model + 1.0)

    def generate_synthetic_samples(self, model, batch_size):
        z_samples = model.reparameterizer.prior([batch_size,
                                                model.config['latent_size']])
        return model.decode(z_samples)

    def _augment_data(self, x):
        ''' return batch_size worth of samples that are augmented
            from the teacher model '''
        if self.ratio == 1.0:
            return x            # base case

        batch_size = x.size(0)
        num_teacher_samples = int(batch_size * self.ratio)
        num_student_samples = max(batch_size - num_teacher_samples, 1)
        generated_teacher_samples = self.generate_synthetic_samples(self.teacher, batch_size)
        return torch.cat([x[0:num_student_samples],
                          generated_teacher_samples[0:num_teacher_samples]], 0)

    def forward(self, x):
        x_augmented = self._augment_data(x)
        x_recon_student, params_student = self.student(x_augmented)
        if self.teacher is not None:
            # return the student and teacher params
            # only teacher Q(z|x) is needed, so dont run decode step
            z_logits_teacher = self.teacher.encode(x)
            _, params_teacher = self.teacher.reparameterize(z_logits_teacher)
            detach_from_graph(params_teacher)
            return {
                'teacher': {
                    'params': params_teacher
                },
                'student':{
                    'params': params_student,
                    'x_reconstr': x_recon_student
                }
            }

        # return only student outputs
        # this happens when there is no teacher
        # i.e: 0th model
        return {
                'student':{
                    'params': params_student,
                    'x_reconstr': x_recon_student
                }
        }
