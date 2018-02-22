from __future__ import print_function
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
from copy import deepcopy

from models.vae import VAE
from helpers.utils import expand_dims, long_type, squeeze_expand_dim, \
    ones_like, float_type, pad, invert_shuffle, one_hot_np, \
    zero_pad_smaller_cat, check_or_create_dir


def detach_from_graph(param_map):
    for _, v in param_map.items():
        if isinstance(v, dict):
            detach_from_graph(v)
        else:
            v = v.detach_()


def kl_categorical_categorical(dist_a, dist_b, rnd_perm, from_index=0, cuda=False):
    # invert the shuffle for the KL calculation
    # dist_a_logits = invert_shuffle(dist_a['logits'], rnd_perm) # undo the random perm
    # dist_b_logits = invert_shuffle(dist_b['logits'], rnd_perm) # undo the random perm
    dist_a_logits, dist_b_logits = dist_a['logits'], dist_b['logits']

    # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/distributions/python/ops/categorical.py
    dist_a_log_softmax = F.log_softmax(dist_a_logits[from_index:], dim=-1)
    dist_a_softmax = F.softmax(dist_a_logits[from_index:], dim=-1)
    dist_b_log_softmax = F.log_softmax(dist_b_logits[from_index:], dim=-1)

    # zero pad the smaller categorical
    dist_a_log_softmax, dist_b_log_softmax \
        = zero_pad_smaller_cat(dist_a_log_softmax,
                               dist_b_log_softmax,
                               cuda=cuda)
    dist_a_softmax, dist_b_log_softmax \
        = zero_pad_smaller_cat(dist_a_softmax,
                               dist_b_log_softmax,
                               cuda=cuda)

    delta_log_probs1 = dist_a_log_softmax - dist_b_log_softmax
    return torch.sum(dist_a_softmax * delta_log_probs1, dim=-1)


def kl_isotropic_gauss_gauss(dist_a, dist_b, rnd_perm=None):
    n0 = D.Normal(dist_a['mu'], dist_a['logvar'])
    n1 = D.Normal(dist_b['mu'], dist_b['logvar'])
    return torch.sum(D.kl_divergence(n0, n1), dim=-1)


def lazy_generate_modules(model, img_shp, batch_size, cuda):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    data = float_type(cuda)(batch_size, *img_shp).normal_()
    model(Variable(data))


class StudentTeacher(nn.Module):
    def __init__(self, initial_model, **kwargs):
        ''' Helper to keep the student-teacher architecture '''
        super(StudentTeacher, self).__init__()
        self.teacher = None
        self.student = initial_model
        self.current_model = 0
        self.ratio = 1.0
        self.rnd_perm = None
        self.num_teacher_samples = None
        self.num_student_samples = None

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def load(self):
        # load the model if it exists
        if os.path.isdir(".models"):
            model_filename = os.path.join(".models", self.get_name() + ".th")
            if os.path.isfile(model_filename):
                print("loading existing student-teacher model")
                lazy_generate_modules(self, self.student.input_shape,
                                      self.config['batch_size'],
                                      self.config['cuda'])
                self.load_state_dict(torch.load(model_filename))
                return True

        return False

    def save(self, overwrite=False):
        # save the model if it doesnt exist
        check_or_create_dir(".models")
        model_filename = os.path.join(".models", self.get_name() + ".th")
        if not os.path.isfile(model_filename) or overwrite:
            print("saving existing student-teacher model")
            torch.save(self.state_dict(), model_filename)

    def get_name(self):
        #teacher_name = "teacher_" + self.teacher.get_name() if self.teacher else "teacher_none"
        #return teacher_name + "student_" + self.student.get_name()
        return str(self.config['uid']) + str(self.current_model) + self.student.get_name()

    def posterior_regularizer(self, q_z_given_x_t, q_z_given_x_s):
        ''' Evaluates KL(Q_{\Phi})(z | \hat{x}) || Q_{\phi})(z | \hat{x})) '''
        # TF: kl = self.kl_categorical(p=self.q_z_s_given_x_t, q=self.q_z_t_given_x_t)
        if 'discrete' in q_z_given_x_s and 'discrete' in q_z_given_x_t:
            return kl_categorical_categorical(q_z_given_x_s['discrete'],
                                              q_z_given_x_t['discrete'],
                                              self.rnd_perm,
                                              from_index=self.num_student_samples,
                                              cuda=self.config['cuda'])
        elif 'gaussian' in q_z_given_x_s and 'gaussian' in q_z_given_x_t:
            # gauss kl-kl doesnt have any from-index
            return kl_isotropic_gauss_gauss(q_z_given_x_s['gaussian'],
                                            q_z_given_x_t['gaussian'])
        else:
            raise NotImplementedError("unknown distribution requested for kl")

    def _lifelong_loss_function(self, output_map):
        ''' returns a combined loss of the VAE loss
            + regularizers '''
        vae_loss = self.student.loss_function(output_map['student']['x_reconstr_logits'],
                                              output_map['augmented']['data'],
                                              output_map['student']['params'])
        if 'teacher' in output_map and not self.config['disable_regularizers']:
            posterior_regularizer = self.posterior_regularizer(output_map['teacher']['params'],
                                                               output_map['student']['params'])
            diff = int(np.abs(vae_loss['loss'].size(0) - posterior_regularizer.size(0)))
            posterior_regularizer = pad(posterior_regularizer,
                                        diff,
                                        dim=0,
                                        prepend=True,
                                        cuda=self.config['cuda'])
            # posterior_regularizer = posterior_regularizer[self.rnd_perm]
            # posterior_regularizer = invert_shuffle(posterior_regularizer, self.rnd_perm)
            vae_loss['loss_mean'] = torch.mean(vae_loss['loss'] + posterior_regularizer)
            vae_loss['posterior_regularizer_mean'] = torch.mean(posterior_regularizer)

        return vae_loss

    def _ewc(self, fisher_matrix, gamma=30):
        losses = []
        for (nt, pt), (ns, ps), (nf, fish) in zip(self.teacher.named_parameters(),
                                                  self.student.named_parameters(),
                                                  fisher_matrix.items()):
            # print("f {} * (t {} - s {})".format(nf, nt, ns))
            # print("f {} * (t {} - s {})".format(fish.size(), pt.size(), ps.size()))
            # print("f {} * (t {} - s {})".format(type(fish), type(pt), type(ps)))
            if pt.size() == ps.size():
                losses.append(torch.sum(fish * (pt - ps)**2))

        return (gamma / 2.0) * sum(losses)


    def _ewc_loss_function(self, output_map, fisher_matrix):
        ''' returns a combined loss of the VAE loss + EWC '''
        vae_loss = self.student.loss_function(output_map['student']['x_reconstr_logits'],
                                              output_map['augmented']['data'],
                                              output_map['student']['params'])
        if 'teacher' in output_map and fisher_matrix is not None:
            ewc = self._ewc(fisher_matrix)
            vae_loss['ewc_mean'] = ewc
            vae_loss['loss_mean'] = torch.mean(vae_loss['loss']) + ewc

        return vae_loss


    def loss_function(self, output_map, fisher=None):
        if self.config['ewc']:
            return self._ewc_loss_function(output_map, fisher)

        return self._lifelong_loss_function(output_map)

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
        # copy the old student into the teacher
        config_copy = deepcopy(self.student.config)
        config_copy['discrete_size'] += 1
        self.teacher = deepcopy(self.student)

        # create a new student
        self.student = VAE(input_shape=self.teacher.input_shape,
                           num_current_model=self.current_model+1,
                           **{'kwargs': config_copy}
        )

        # forward pass once to build lazy modules
        data = float_type(self.config['cuda'])(self.student.config['batch_size'],
                                               *self.student.input_shape).normal_()
        self.student(Variable(data))

        # copy teacher params into student while
        # omitting the projection weights
        self.teacher, self.student \
            = self.copy_model(self.teacher, self.student, disable_dst_grads=False)

        # update the current model's ratio
        self.current_model += 1
        self.ratio = self.current_model / (self.current_model + 1.0)
        num_teacher_samples = int(self.config['batch_size'] * self.ratio)
        num_student_samples = max(self.config['batch_size'] - num_teacher_samples, 1)
        print("#teacher_samples: ", num_teacher_samples,
              " | #student_samples: ", num_student_samples)

    def generate_synthetic_samples(self, model, batch_size):
        z_samples = model.reparameterizer.prior([batch_size,
                                                 model.reparameterizer.output_size])
        return model.nll_activation(model.decode(z_samples))

    def generate_synthetic_sequential_samples(self, model, num_rows=8):
        assert self.config['reparam_type'] == 'mixture' \
            or self.config['reparam_type'] == 'discrete'

        # create a grid of one-hot vectors for displaying in visdom
        # uses one row for original dimension of discrete component
        num_current_discrete = model.reparameterizer.config['discrete_size']
        num_orig_discrete = num_current_discrete - model.num_current_model
        discrete_indices = np.asarray([[np.random.randint(0, num_orig_discrete)]
                                       for _ in range(num_rows)]).reshape(-1, num_rows)
        if model.num_current_model > 0:
            extra_discrete_indices = np.asarray([[i]*num_rows for i in range(num_orig_discrete,
                                                                             num_current_discrete)])
            discrete_indices = np.vstack([discrete_indices, extra_discrete_indices])

        discrete_indices = discrete_indices.reshape(-1)
        with torch.no_grad():
            z_samples = Variable(torch.from_numpy(one_hot_np(num_current_discrete,
                                                             discrete_indices)))
            z_samples = z_samples.type(float_type(self.config['cuda']))

            if self.config['reparam_type'] == 'mixture':
                ''' add in the gaussian prior '''
                z_gauss = model.reparameterizer.gaussian.prior(
                    [z_samples.size(0), model.reparameterizer.gaussian.output_size]
                )
                z_samples = torch.cat([z_gauss, z_samples], dim=-1)

            return model.nll_activation(model.decode(z_samples))

    def _augment_data(self, x):
        ''' return batch_size worth of samples that are augmented
            from the teacher model '''
        if self.ratio == 1.0 or not self.training: #or self.config['ewc']:
            return x   # base case

        batch_size = x.size(0)
        self.num_teacher_samples = int(batch_size * self.ratio)
        self.num_student_samples = max(batch_size - self.num_teacher_samples, 1)
        generated_teacher_samples = self.generate_synthetic_samples(self.teacher, batch_size)
        merged =  torch.cat([x[0:self.num_student_samples],
                             generated_teacher_samples[0:self.num_teacher_samples]], 0)

        # workaround for batchnorm on multiple GPUs
        # we shuffle the data and unshuffle it later for
        # the posterior regularizer
        # self.rnd_perm = torch.randperm(merged.size(0))
        # if self.config['cuda']:
        #     self.rnd_perm = self.rnd_perm.cuda()

        # return merged[self.rnd_perm]

        return merged

    def forward(self, x):
        x_augmented = self._augment_data(x).contiguous()
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
            self.teacher.eval()
            z_logits_teacher = self.teacher.encode(x_augmented)
            _, params_teacher = self.teacher.reparameterize(z_logits_teacher)
            # detach_from_graph(params_teacher)
            ret_map['teacher']= {
                'params': params_teacher
            }

        return ret_map
