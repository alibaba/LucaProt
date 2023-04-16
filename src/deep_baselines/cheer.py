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
@email: sanyuan.**alibaba-inc.com
@tel: 137****6540
@datetime: 2023/3/28 15:45
@project: DeepProtFunc
@file: cheer
@desc: CHEER: HierarCHical taxonomic classification for viral mEtagEnomic data via deep learning
'''
import logging
import sys
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.autograd import Variable
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
sys.path.append("../../src/common")
try:
    from loss import *
    from utils import *
    from multi_label_metrics import *
    from metrics import *
except ImportError:
    from src.common.loss import *
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *

logger = logging.getLogger(__name__)


class CatWCNN(nn.Module):
    def __init__(self, config, args):
        super(CatWCNN, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.embedding_trainable = config.embedding_trainable
        self.max_position_embeddings = config.max_position_embeddings
        self.num_labels = config.num_labels
        self.channel_in = args.channel_in
        self.kernel_nums = config.kernel_nums
        self.kernel_sizes = config.kernel_sizes
        self.dropout = config.dropout
        self.bias = config.bias
        self.fc_size1 = config.fc_size1
        self.fc_size2 = config.fc_size2
        self.padding_idx = config.padding_idx
        self.output_mode = args.output_mode
        if self.num_labels == 2:
            self.num_labels = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        if self.embedding_trainable:
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        self.convs3 = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        self.convs4 = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        self.convs5 = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        self.convs6 = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(sum(self.kernel_nums)*self.channel_in, self.fc_size1, bias=self.bias)
        self.fc2 = nn.Linear(self.fc_size1, self.fc_size2, bias=self.bias)
        self.linear_layer = nn.Linear(self.fc_size2, self.num_labels, bias=self.bias)

        if args.sigmoid:
            self.output = nn.Sigmoid()
        else:
            if self.num_labels > 1:
                self.output = nn.Softmax(dim=1)

            else:
                self.output = None

        self.loss_type = args.loss_type

        # positive weight
        if hasattr(config, "pos_weight"):
            self.pos_weight = config.pos_weight
        elif hasattr(args, "pos_weight"):
            self.pos_weight = args.pos_weight
        else:
            self.pos_weight = None

        if hasattr(config, "weight"):
            self.weight = config.weight
        elif hasattr(args, "weight"):
            self.weight = args.weight
        else:
            self.weight = None

        if self.output_mode in ["regression"]:
            self.loss_fct = MSELoss()
        elif self.output_mode in ["multi_label", "multi-label"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [1, 1, 1, ,1, 1...] length: self.num_labels
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == self.num_labels
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss(reduction=config.loss_reduction if hasattr(config, "loss_reduction") else "sum")
            elif self.loss_type == "asl":
                self.loss_fct = AsymmetricLossOptimized(gamma_neg=args.asl_gamma_neg if hasattr(args, "asl_gamma_neg") else 4,
                                                        gamma_pos=args.asl_gamma_pos if hasattr(args, "asl_gamma_pos") else 1,
                                                        clip=args.clip if hasattr(args, "clip") else 0.05,
                                                        eps=args.eps if hasattr(args, "eps") else 1e-8,
                                                        disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss if hasattr(args, "disable_torch_grad_focal_loss") else False)
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
            elif self.loss_type == "multilabel_cce":
                self.loss_fct = MultiLabel_CCE(normalization=False)
        elif self.output_mode in ["binary_class", "binary-class"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [0.9]
                    if isinstance(self.pos_weight, int):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.long).to(args.device)
                    elif isinstance(self.pos_weight, float):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32).to(args.device)
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == 1
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss()
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
        elif self.output_mode in ["multi_class", "multi-class"]:
            if self.weight:
                # [1, 1, 1, ,1, 1...] length: self.num_labels
                assert self.weight.ndim == 1 and self.weight.shape[0] == self.num_labels
                self.loss_fct = CrossEntropyLoss(weight=self.weight)
            else:
                self.loss_fct = CrossEntropyLoss()
        else:
            raise Exception("Not support output mode: %s." % self.output_mode)

    def forward(self, x, lengths=None, labels=None):
        B, L = x.shape
        x = self.embedding(x)
        # B, Len, dim
        x = x.reshape((B, self.channel_in, -1, self.embedding_dim))
        x1 = x[:, 0, :, :].reshape(B, 1, -1, self.embedding_dim)
        x2 = x[:, 1, :, :].reshape(B, 1, -1, self.embedding_dim)
        x3 = x[:, 2, :, :].reshape(B, 1, -1, self.embedding_dim)
        x4 = x[:, 3, :, :].reshape(B, 1, -1, self.embedding_dim)
        x5 = x[:, 4, :, :].reshape(B, 1, -1, self.embedding_dim)
        x6 = x[:, 5, :, :].reshape(B, 1, -1, self.embedding_dim)
        #
        x1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs1]
        x2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        x3 = [F.relu(conv(x3)).squeeze(3) for conv in self.convs3]
        x4 = [F.relu(conv(x4)).squeeze(3) for conv in self.convs4]
        x5 = [F.relu(conv(x5)).squeeze(3) for conv in self.convs5]
        x6 = [F.relu(conv(x6)).squeeze(3) for conv in self.convs6]

        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x1]
        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x2]
        x3 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x3]
        x4 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x4]
        x5 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x5]
        x6 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x6]

        x1 = torch.cat(x1, 1)
        x2 = torch.cat(x2, 1)
        x3 = torch.cat(x3, 1)
        x4 = torch.cat(x4, 1)
        x5 = torch.cat(x5, 1)
        x6 = torch.cat(x6, 1)

        x = torch.cat((x1, x2, x3, x4, x5, x6), 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.linear_layer(x)

        if self.output:
            output = self.output(logits)
        else:
            output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            outputs = [loss, *outputs]

        return outputs


class WDCNN(nn.Module):
    def __init__(self, config, args):
        super(WDCNN, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.embedding_trainable = config.embedding_trainable
        self.max_position_embeddings = config.max_position_embeddings
        self.num_labels = config.num_labels
        self.channel_in = args.channel_in
        self.kernel_nums = config.kernel_nums
        self.kernel_sizes = config.kernel_sizes
        self.dropout = config.dropout
        self.bias = config.bias
        self.fc_size = config.fc_size
        self.padding_idx = config.padding_idx
        self.output_mode = args.output_mode
        if self.num_labels == 2:
            self.num_labels = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        if self.embedding_trainable:
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channel_in, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(self.kernel_nums[i], self.kernel_nums[i], (kernel_size, 1), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(sum(self.kernel_nums), self.fc_size)
        self.linear_layer = nn.Linear(self.fc_size, self.num_labels)

        if args.sigmoid:
            self.output = nn.Sigmoid()
        else:
            if self.num_labels > 1:
                self.output = nn.Softmax(dim=1)

            else:
                self.output = None

        self.loss_type = args.loss_type

        # positive weight
        if hasattr(config, "pos_weight"):
            self.pos_weight = config.pos_weight
        elif hasattr(args, "pos_weight"):
            self.pos_weight = args.pos_weight
        else:
            self.pos_weight = None

        if hasattr(config, "weight"):
            self.weight = config.weight
        elif hasattr(args, "weight"):
            self.weight = args.weight
        else:
            self.weight = None

        if self.output_mode in ["regression"]:
            self.loss_fct = MSELoss()
        elif self.output_mode in ["multi_label", "multi-label"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [1, 1, 1, ,1, 1...] length: self.num_labels
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == self.num_labels
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss(reduction=config.loss_reduction if hasattr(config, "loss_reduction") else "sum")
            elif self.loss_type == "asl":
                self.loss_fct = AsymmetricLossOptimized(gamma_neg=args.asl_gamma_neg if hasattr(args, "asl_gamma_neg") else 4,
                                                        gamma_pos=args.asl_gamma_pos if hasattr(args, "asl_gamma_pos") else 1,
                                                        clip=args.clip if hasattr(args, "clip") else 0.05,
                                                        eps=args.eps if hasattr(args, "eps") else 1e-8,
                                                        disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss if hasattr(args, "disable_torch_grad_focal_loss") else False)
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
            elif self.loss_type == "multilabel_cce":
                self.loss_fct = MultiLabel_CCE(normalization=False)
        elif self.output_mode in ["binary_class", "binary-class"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [0.9]
                    if isinstance(self.pos_weight, int):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.long).to(args.device)
                    elif isinstance(self.pos_weight, float):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32).to(args.device)
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == 1
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss()
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
        elif self.output_mode in ["multi_class", "multi-class"]:
            if self.weight:
                # [1, 1, 1, ,1, 1...] length: self.num_labels
                assert self.weight.ndim == 1 and self.weight.shape[0] == self.num_labels
                self.loss_fct = CrossEntropyLoss(weight=self.weight)
            else:
                self.loss_fct = CrossEntropyLoss()
        else:
            raise Exception("Not support output mode: %s." % self.output_mode)

    def forward(self, x, lengths=None, labels=None):
        B, L = x.shape
        x = self.embedding(x)
        x = x.reshape((B, self.channel_in, -1, self.embedding_dim))
        x = [F.relu(conv(x)) for conv in self.convs1]
        x = [F.relu(conv(i)).squeeze(3) for conv, i in zip(self.convs2, x)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.linear_layer(x)

        if self.output:
            output = self.output(logits)
        else:
            output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            outputs = [loss, *outputs]

        return outputs


class WCNN(nn.Module):
    def __init__(self, config, args):
        super(WCNN, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.embedding_trainable = config.embedding_trainable
        self.max_position_embeddings = config.max_position_embeddings
        self.num_labels = config.num_labels
        self.channel_in = args.channel_in
        self.kernel_nums = config.kernel_nums
        self.kernel_sizes = config.kernel_sizes
        self.dropout = config.dropout
        self.bias = config.bias
        self.fc_size = config.fc_size
        self.padding_idx = config.padding_idx
        self.output_mode = args.output_mode
        if self.num_labels == 2:
            self.num_labels = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        if self.embedding_trainable:
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False
            self.embedding .weight[self.padding_idx] = torch.zeros(self.embedding_dim, device=args.device, requires_grad=False)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channel_in, self.kernel_nums[i], (kernel_size, self.embedding_dim), bias=self.bias) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(sum(self.kernel_nums), self.fc_size)
        self.linear_layer = nn.Linear(self.fc_size, self.num_labels)

        if args.sigmoid:
            self.output = nn.Sigmoid()
        else:
            if self.num_labels > 1:
                self.output = nn.Softmax(dim=1)

            else:
                self.output = None

        self.loss_type = args.loss_type

        # positive weight
        if hasattr(config, "pos_weight"):
            self.pos_weight = config.pos_weight
        elif hasattr(args, "pos_weight"):
            self.pos_weight = args.pos_weight
        else:
            self.pos_weight = None

        if hasattr(config, "weight"):
            self.weight = config.weight
        elif hasattr(args, "weight"):
            self.weight = args.weight
        else:
            self.weight = None

        if self.output_mode in ["regression"]:
            self.loss_fct = MSELoss()
        elif self.output_mode in ["multi_label", "multi-label"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [1, 1, 1, ,1, 1...] length: self.num_labels
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == self.num_labels
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss(reduction=config.loss_reduction if hasattr(config, "loss_reduction") else "sum")
            elif self.loss_type == "asl":
                self.loss_fct = AsymmetricLossOptimized(gamma_neg=args.asl_gamma_neg if hasattr(args, "asl_gamma_neg") else 4,
                                                        gamma_pos=args.asl_gamma_pos if hasattr(args, "asl_gamma_pos") else 1,
                                                        clip=args.clip if hasattr(args, "clip") else 0.05,
                                                        eps=args.eps if hasattr(args, "eps") else 1e-8,
                                                        disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss if hasattr(args, "disable_torch_grad_focal_loss") else False)
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
            elif self.loss_type == "multilabel_cce":
                self.loss_fct = MultiLabel_CCE(normalization=False)
        elif self.output_mode in ["binary_class", "binary-class"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [0.9]
                    if isinstance(self.pos_weight, int):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.long).to(args.device)
                    elif isinstance(self.pos_weight, float):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32).to(args.device)
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == 1
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss()
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 1,
                                          gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 0.25,
                                          normalization=False,
                                          reduce=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
        elif self.output_mode in ["multi_class", "multi-class"]:
            if self.weight:
                # [1, 1, 1, ,1, 1...] length: self.num_labels
                assert self.weight.ndim == 1 and self.weight.shape[0] == self.num_labels
                self.loss_fct = CrossEntropyLoss(weight=self.weight)
            else:
                self.loss_fct = CrossEntropyLoss()
        else:
            raise Exception("Not support output mode: %s." % self.output_mode)

    def forward(self, x, lengths=None, labels=None):
        B, L = x.shape
        x = self.embedding(x)
        x = x.reshape((B, self.channel_in, -1, self.embedding_dim))
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.linear_layer(x)

        if self.output:
            output = self.output(logits)
        else:
            output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            outputs = [loss, *outputs]

        return outputs


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def seq_encode(seq, channel_in, max_len, trunc_type, vocab):
    seq_len = len(seq)
    if seq_len >= max_len:
        actural_len = max_len
        if trunc_type == "right":
            processed_seq = list(seq[:max_len])
        else:
            processed_seq = list(seq[-max_len:])
    else:
        actural_len = len(seq)
        max_len_per_segment = max_len//channel_in
        segment_seq_list = []
        real_len_per_segment = (len(seq) + channel_in - 1)//channel_in
        for idx in range(channel_in):
            real_segment = list(seq[idx * real_len_per_segment: min(idx * real_len_per_segment + real_len_per_segment, seq_len)])
            if len(real_segment) < max_len_per_segment:
                real_segment += ["[PAD]"] * (max_len_per_segment - len(real_segment) )
            segment_seq_list.append(real_segment)
        processed_seq = [item for sublist in segment_seq_list for item in sublist]
    processed_seq_id = []
    for char in processed_seq:
        processed_seq_id.append(vocab[char])
    return processed_seq_id, actural_len