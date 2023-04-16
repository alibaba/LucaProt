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
@datetime: 2023/2/28 16:36
@project: DeepProtFunc
@file: virhunter
@desc: VirHunter: A Deep Learning-Based Method for Detection of Novel RNA Viruses in Plant Sequencing Data
'''
import logging
import sys
import torch
from torch.nn.functional import one_hot
import torch.nn.functional
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
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


class VirHunter(nn.Module):
    def __init__(self, config, args):
        super(VirHunter, self).__init__()
        self.one_hot_encode = config.one_hot_encode
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.embedding_trainable = config.embedding_trainable
        self.embedding_dim = config.embedding_dim
        self.num_labels = config.num_labels
        self.kernel_num = config.kernel_num
        self.kernel_size = config.kernel_size
        self.dropout = config.dropout
        self.reverse = config.reverse
        self.bias = config.bias
        self.fc_size = config.fc_size
        self.output_mode = args.output_mode
        self.padding_idx = config.padding_idx
        if self.num_labels == 2:
            self.num_labels = 1
        if not self.one_hot_encode:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
            if self.embedding_trainable:
                self.embedding.weight.requires_grad = True
            else:
                self.embedding.weight.requires_grad = False

        self.hidden_layers = nn.ModuleList(
            [
                nn.Conv1d(self.vocab_size if self.one_hot_encode else self.embedding_dim, self.kernel_num, self.kernel_size, bias=self.bias),
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool1d(self.max_position_embeddings - self.kernel_size + 1, stride=1),
                nn.Dropout(self.dropout)
            ]
        )
        if self.reverse:
            self.dense = nn.Linear(self.kernel_num + self.kernel_num, self.fc_size, bias=self.bias)
        else:
            self.dense = nn.Linear(self.kernel_num, self.fc_size, bias=self.bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_layer = nn.Linear(self.fc_size, self.num_labels, bias=self.bias)

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

    def forward(self, x, reverse_x=None, lengths=None, labels=None):
        if not self.one_hot_encode:
            x = self.embedding(x)
        else:
            x = one_hot(x, num_classes=self.vocab_size).to(torch.float32)
        x = x.permute(0, 2, 1)
        if reverse_x:
            for layer in self.hidden_layers:
                x = layer(x)
            for layer in self.hidden_layers:
                reverse_x = layer(reverse_x)
            x = torch.cat([x, reverse_x], dim=-1)
        else:
            for layer in self.hidden_layers:
                x = layer(x)
        x = torch.squeeze(x, -1)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
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


def seq_encode(seq, max_len, trunc_type, vocab):
    seq_len = len(seq)
    if seq_len >= max_len:
        actural_len = max_len
        if trunc_type == "right":
            processed_seq = list(seq[:max_len])
        else:
            processed_seq = list(seq[-max_len:])
    else:
        actural_len = len(seq)
        processed_seq = list(seq) + ["[PAD]"] * (max_len - seq_len)
    processed_seq_id = []
    for char in processed_seq:
        processed_seq_id.append(vocab[char])
    return processed_seq_id, actural_len


def one_hot_encode(seq, max_len, trunc_type, vocab):
    processed_seq_id, actural_len = seq_encode(seq, max_len, trunc_type, vocab)
    one_hot = []
    for idx in processed_seq_id:
        cur_one_hot = [0.0] * len(vocab)
        cur_one_hot[idx] = 1.0
        one_hot.append(cur_one_hot)
    return one_hot, actural_len


if __name__ == "__main__":
   # protein_list = ["[PAD]", "I", "M", "T", "N", "K", "S", "R", "L", "P", "H", "Q", "V", "A", "D", "E", "G", "S", "F", "Y", "W", "C", "O"]
    protein_list = ["[PAD]", "A", "T", "C", "G"]
    int_to_protein = {idx: char for idx, char in enumerate(protein_list)}
    protein_to_int = {char: idx for idx, char in int_to_protein.items()}

    print(one_hot_encode("AATCG", 6, protein_to_int))