#!/usr/bin/env python
# encoding: utf-8
'''
*Copyright (c) 2023, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.

@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/19 14:46
@project: DeepProtFunc
@file: layers
@desc: layers for GCN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        :param h: (batch_size, N, in_features)
        :param adj: (batch_size, N, N)
        :return:
        '''
        # (Batch_size, N, in_features) * (in_features, out_features) -> (N, out_features)
        h = self.W(h)
        batch_size, N, _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[1]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, h, adj):
        '''
        :param input: (batch_size, N, dim)
        :param adj: (batch_size, N, N)
        :return:
        '''
        dv = 'cuda' if h.is_cuda else 'cpu'
        batch_size = h.size()[0]
        input_list = torch.split(h, 1, dim=0)
        result_list = [None] * batch_size
        for idx, input in enumerate(input_list):
            input = input.squeeze(0)
            N = input.size()[0]
            edge = adj[idx, :, :].squeeze(0).nonzero().t()  # non zero edge [2， 52]
            h = torch.mm(input, self.W) # 2d matrix multiplication
            h[h != h] = 0
            # h: N x out
            assert not torch.isnan(h).any()

            # Self-attention on the nodes - Shared attention mechanism
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
            # edge: 2*D x E

            edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
            edge_e[edge_e != edge_e] = 0
            assert not torch.isnan(edge_e).any()
            # edge_e: E

            O_e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
            # e_rowsum: N x 1

            nonzeros = torch.nonzero(O_e_rowsum)
            e_rowsum = torch.ones(size=(N, 1), device=dv)
            e_rowsum[nonzeros[:, 0]] = O_e_rowsum[nonzeros[:, 0]]

            edge_e = self.dropout(edge_e)
            # edge_e: E

            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)  # [22, 3*256]
            h_prime[h_prime != h_prime] = 0
            assert not torch.isnan(h_prime).any()
            # h_prime: N x out

            h_prime = h_prime.div(e_rowsum)
            h_prime[h_prime != h_prime] = 0
            # h_prime: N x out
            assert not torch.isnan(h_prime).any()

            if self.concat:
                # if this layer is not last layer,
                results = F.elu(h_prime)
            else:
                # if this layer is last layer,
                results = h_prime # [22, 3 * 256]
            result_list[idx] = results.unsqueeze(0)
        results = torch.cat(result_list, dim=0)
        return results

    def __repr__(self):
            return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

