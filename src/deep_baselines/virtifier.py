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
@datetime: 2023/2/28 17:17
@project: DeepProtFunc
@file: virtifier
@desc: Virtifier: a deep learning-based identifier for viral sequences from metagenomes
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


class Virtifier(nn.Module):
    def __init__(self, config, args):
        '''
        :param config:
        :param args:
        '''
        super(Virtifier, self).__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.embedding_init = config.embedding_init
        self.embedding_init_path = config.embedding_init_path
        self.embedding_trainable = config.embedding_trainable
        self.embedding_dim = config.embedding_dim
        self.bidirectional = config.bidirectional
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.bias = config.bias
        self.fc_size = config.fc_size
        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        self.padding_idx = config.padding_idx
        self.batch_first = config.batch_first
        self.batch_norm = config.batch_norm
        self.use_last = config.use_last
        self.rnn_model = config.rnn_model
        if hasattr(config, "padding_idx"):
            self.padding_idx = config.padding_idx
        elif hasattr(args, "padding_idx"):
            self.padding_idx = args.padding_idx
        else:
            self.padding_idx = 0
        if self.num_labels == 2:
            self.num_labels = 1

        if self.embedding_init:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embedding_init_path).float())
            if self.embedding_trainable:
                self.embedding.weight.requires_grad = True
            else:
                self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)

        if self.num_layers == 1:
            if self.rnn_model.lower() == 'lstm':
                self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                                    num_layers=self.num_layers, bidirectional=self.bidirectional,
                                    batch_first=self.batch_first)
            elif self.rnn_model.lower() == 'gru':
                self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                                    num_layers=self.num_layers, bidirectional=self.bidirectional,
                                    batch_first=self.batch_first)

        else:
            if self.rnn_model.lower() == 'lstm':
                self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                                   num_layers=self.num_layers, bidirectional=self.bidirectional,
                                   batch_first=self.batch_first,
                                   dropout=self.dropout)
            elif self.rnn_model.lower() == 'gru':
                self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers, bidirectional=self.bidirectional,
                                  batch_first=self.batch_first,
                                  dropout=self.dropout)
        self.bn = None
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim)
        self.linear_layer = nn.Linear(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim, self.num_labels, bias=self.bias)

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

    def forward(self, x, lengths, labels=None):
        x_embed = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x_embed, lengths.to('cpu'), enforce_sorted=False, batch_first=self.batch_first)
        packed_output, ht = self.rnn(packed_input, None)
        out_rnn, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        if self.use_last:
            row_indices = torch.arange(0, out_rnn.size(0)).long()
            col_indices = lengths - 1
            if next(self.parameters()).is_cuda:
                row_indices = row_indices.cuda()
                col_indices = col_indices.cuda()
            last_tensor = out_rnn[row_indices, col_indices, :]
        else:
            # use mean
            mask = x != self.padding_idx
            max_len = torch.max(lengths)
            col_indices = torch.arange(0, max_len).long()
            if next(self.parameters()).is_cuda:
                col_indices = col_indices.cuda()
            mask = mask[:, col_indices]
            denom = torch.sum(mask, -1, keepdim=True)
            last_tensor = torch.sum(out_rnn * mask.unsqueeze(-1), dim=1) / denom

        if self.bn:
            last_tensor = self.bn(last_tensor)
        logits = self.linear_layer(last_tensor)
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