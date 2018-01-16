import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

from helpers.utils import to_data, softmax_accuracy, expand_dims, \
    int_type, float_type, long_type, add_weight_norm


class RelationalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, cuda=False, ngpu=1):
        super(RelationalNetwork, self).__init__()
        self.use_cuda = cuda
        self.ngpu = ngpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # build the final combined projector module
        self.proj = self._build_proj_model()

    def _build_proj_model(self):
        proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

        if self.ngpu > 1:
            proj = nn.DataParallel(proj)

        if self.use_cuda:
            proj = proj.cuda()

        return proj

    def _lazy_generate_rn(self, input_size, latent_size, output_size):
        if not hasattr(self, 'rn'):
            self.rn = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.BatchNorm2d(latent_size),  # Input: :math:`(N, C, H, W)`
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                #nn.BatchNorm1d(self.hidden_size),
                nn.BatchNorm2d(latent_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, output_size),
                nn.ReLU()
            )

            if self.ngpu > 1:
                self.rn = nn.DataParallel(self.rn)

            if self.use_cuda:
                self.rn = self.rn.cuda()

    def forward(self, tuples):
        rn_buffer = [] # container to hold the tuples

        for i in range(len(tuples)):
            chunk_i = tuples[i].contiguous()

            # for j in range(i+1, num_chans):
            for j in range(len(tuples)):
                chunk_j = tuples[j].contiguous()
                merged = torch.cat([chunk_i, chunk_j], -1)
                rn_buffer.append(expand_dims(merged, 1))

        # aggregate the buffer
        rn_buffer = torch.cat(rn_buffer, 1)

        # generate the relational network lazily
        self._lazy_generate_rn(input_size=rn_buffer.size(-1),
                               latent_size=rn_buffer.size(1),  # for BN
                               output_size=self.hidden_size)

        # squeeze, reduce over the concatenations and project
        rbo = torch.squeeze(self.rn(rn_buffer))
        rn_output = torch.sum(rbo, 1)
        return self.proj(rn_output)
