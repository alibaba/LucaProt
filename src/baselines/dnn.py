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
@datetime: 2022/12/29 19:36
@project: DeepProtFunc
@file: dnn
@desc: DNN (based protein structure embeddding (matrix or vector) for classification
'''
import os, sys
import argparse
import logging
import time, json
import shutil

import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from tensorboardX import SummaryWriter
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from common.loss import AsymmetricLossOptimized, FocalLoss, MultiLabel_CCE
    from utils import *
    from multi_label_metrics import *
    from metrics import *
except ImportError:
    from src.common.loss import AsymmetricLossOptimized, FocalLoss, MultiLabel_CCE
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="input dir, including *.csv/*.txt.")
    parser.add_argument("--separate_file", action="store_true", help="The id of each sample in the dataset is separate from its details")
    parser.add_argument("--tfrecords", action="store_true", help="whether the dataset is in tfrecords")
    parser.add_argument("--filename_pattern", default=None, type=str, help="the dataset filename pattern, such as {}_with_pdb_emb.csv including train_with_pdb_emb.csv, dev_with_pdb_emb.csv, and test_with_pdb_emb.csv in ${data_dir}")
    parser.add_argument("--dataset_name", default="rdrp_40_extend", type=str, required=True, help="dataset name")
    parser.add_argument("--dataset_type", default="protein", type=str, required=True, choices=["protein", "dna", "rna"], help="dataset type")
    parser.add_argument("--task_type", default="multi_label", type=str, required=True, choices=["multi_label", "multi_class", "binary_class"], help="task type")
    parser.add_argument("--label_type", default="rdrp", type=str, required=True, help="label type.")
    parser.add_argument("--label_filepath", default=None, type=str, required=True, help="label filepath.")

    parser.add_argument("--output_dir", default="./result/", type=str, required=True, help="output dir.")
    parser.add_argument("--log_dir", default="./logs/", type=str, required=True, help="log dir.")
    parser.add_argument("--tb_log_dir", default="./tb-logs/", type=str, required=True, help="tensorboard log dir.")

    # Other parameters
    parser.add_argument("--config_path", default=None, type=str, required=True, help="the model configuration filepath")
    parser.add_argument("--cache_dir", default="", type=str, help="cache dir")

    parser.add_argument("--do_train", action="store_true", help="whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="whether to evaluation during training at each logging step.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="the initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="if > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", help="overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true", help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="for fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="for distributed training: local_rank")

    # multi-label/ binary-class
    parser.add_argument("--sigmoid", action="store_true", help="classifier add sigmoid if task_type is binary-class or multi-label")

    # loss func
    parser.add_argument("--loss_type",  type=str, default="bce", choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"], help="loss type")

    # which metric for model finalization selected
    parser.add_argument("--max_metric_type",  type=str, default="f1", required=True, choices=["acc", "jaccard", "prec", "recall", "f1", "fmax", "pr_auc", "roc_auc"], help="which metric for model selected")
    parser.add_argument("--early_stopping_rounds", default=None, type=int, help="early stopping rounds.")

    # for focal Loss
    parser.add_argument("--focal_loss_alpha", type=float, default=0.7, help="focal loss alpha value")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0, help="focal loss gamma value")
    parser.add_argument("--focal_loss_reduce", action="store_true", help="mean for one sample(default sum)")

    # for asymmetric Loss
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0, help="negative gamma for asl")
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0, help="positive gamma for asl")

    parser.add_argument("--weight", default=None, type=float,  help="loss weight for multi_class task")
    parser.add_argument("--pos_weight", default=None, type=float,  help="positive weight")
    parser.add_argument("--dropout", default=None, type=float, help="dropout")
    parser.add_argument("--bias", action="store_true", help="bias")

    parser.add_argument("--save_all",  action="store_true", help="save all check-point)")
    parser.add_argument("--delete_old",  action="store_true", help="delete old check-point)")

    args = parser.parse_args()
    args.output_mode = args.task_type
    return args


def args2config(args, config):
    if args.dropout:
        config.dropout = args.dropout
    if args.bias:
        config.bias = args.bias
    if args.pos_weight:
        config.pos_weight = args.pos_weight
    if args.weight:
        config.weight = args.weight
    if args.num_labels:
        config.num_labels = args.num_labels
    if args.task_type in ["binary_class", "binary-class"]:
        config.num_labels = 2


def load_dataset(args, dataset_type):
    '''
    :param args:
    :param dataset_type:
    :return:
    '''
    x = []
    y = []
    if os.path.exists(args.label_filepath):
        label_list = load_labels(args.label_filepath, header=True)
    else:
        label_list = load_labels(os.path.join(args.data_dir, "label.txt"), header=True)
    label_map = {name: idx for idx, name in enumerate(label_list)}
    npz_filpath = os.path.join(args.data_dir, "%s_emb.npz" % dataset_type)
    if os.path.exists(npz_filpath):
        npzfile = np.load(npz_filpath, allow_pickle=True)
        x = npzfile["x"]
        y = npzfile["y"]
    else:
        cnt = 0
        if args.filename_pattern:
            filepath = os.path.join(args.data_dir, args.filename_pattern.format(dataset_type))
        else:
            filepath = os.path.join(args.data_dir, "%s_with_pdb_emb.csv" % dataset_type)
        header = False
        header_filter = False
        if filepath.endswith(".csv"):
            header = True
            header_filter = True
        for row in file_reader(filepath, header=header, header_filter=header_filter):
            prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
            embedding_filepath = os.path.join(args.data_dir, "embs", emb_filename)
            if os.path.exists(embedding_filepath):
                emb = torch.load(embedding_filepath)
                embedding_info = emb["bos_representations"][36].numpy()
                x.append(embedding_info)
                if args.task_type in ["multi-class", "multi_class"]:
                    label = label_map[label]
                elif args.task_type == "regression":
                    label = float(label)
                elif args.task_type in ["multi-label", "multi_label"]:
                    if isinstance(label, str):
                        label = [0] * len(label_map)
                        for label_name in eval(label):
                            label_id = label_map[label_name]
                            label[label_id] = 1
                    else:
                        label = [0] * len(label_map)
                        for label_name in label:
                            label_id = label_map[label_name]
                            label[label_id] = 1
                elif args.task_type in ["binary-class", "binary_class"]:
                    label = label_map[label]
                y.append(label)
                cnt += 1
                if cnt % 10000 == 0:
                    print("done %d" % cnt)
        x = np.array(x)
        y = np.array(y)
        np.savez(npz_filpath, x=x, y=y)
        print("%s: x.shape: %s, y.shape: %s" %(dataset_type, str(x.shape), str(y.shape)))
    return torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)), label_list


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class DNN(nn.Module):
    def __init__(self, config, args):
        super(DNN, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        if config.activate_func == "tanh":
            self.activate = nn.Tanh()
        elif config.activate_func == "relu":
            self.activate = nn.ReLU()
        elif config.activate_func == "gelu":
            self.activate = nn.GELU()
        elif config.activate_func == "leakyrelu":
            self.activate = nn.LeakyReLU()
        else:
            self.activate = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)
        layers = []
        cur_input_dim = config.input_dim
        for idx in range(len(config.hidden_dim_list)):
            cur_output_dim = config.hidden_dim_list[idx]
            layer = nn.Linear(in_features=cur_input_dim, out_features=cur_output_dim, bias=config.bias)
            cur_input_dim = cur_output_dim
            layers.append(layer)
            layers.append(self.dropout)
            layers.append(self.activate)
        output_dim = config.num_labels
        if output_dim == 2:
            output_dim = 1
        layer = nn.Linear(in_features=cur_input_dim, out_features=output_dim, bias=config.bias)
        layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if args.sigmoid:
            self.output = nn.Sigmoid()
        else:
            if self.num_labels > 1:
                self.output = nn.Softmax(dim=1)

            else:
                self.output = None
        # self.net = nn.Sequential(*layers)

        self.loss_type = args.loss_type

        # weight for the loss function
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

    def forward(self, inputs, labels):
        # logits = self.net(inputs)
        logits = inputs
        for i, layer_module in enumerate(self.layers):
            logits = layer_module(logits)
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


def run():
    args = main()
    logging.basicConfig(format="%(asctime)s-%(levelname)s-%(name)s | %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # overwrite the output dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir)
    # create logger dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    # create tensorboard logger dir
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    config_class = BertConfig
    config = config_class(**json.load(open(args.config_path, "r")))

    args2config(args, config)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1 if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    model = DNN(config, args)

    model.to(args.device)

    train_dataset, label_list = load_dataset(args, "train")

    dev_dataset, _ = load_dataset(args, "dev")

    test_dataset, _ = load_dataset(args, "test")

    config.num_labels = len(label_list)
    args.num_labels = config.num_labels

    log_fp.write("Model Config:\n %s\n" % str(config))
    log_fp.write("#" * 50 + "\n")
    log_fp.write("Mode Architecture:\n %s\n" % str(model))
    log_fp.write("#" * 50 + "\n")
    log_fp.write("num labels: %d\n" % args.num_labels)
    log_fp.write("#" * 50 + "\n")

    max_metric_model_info = None
    if args.do_train:
        logger.info("++++++++++++Training+++++++++++++")
        global_step, tr_loss, max_metric_model_info = trainer(args, model, train_dataset, dev_dataset, test_dataset, log_fp=log_fp)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    #  save
    if args.do_train:
        logger.info("++++++++++++Save Model+++++++++++++")
        # Create output directory if needed

        best_output_dir = os.path.join(args.output_dir, "best")

        global_step = max_metric_model_info["global_step"]
        prefix = "checkpoint-{}".format(global_step)
        shutil.copytree(os.path.join(args.output_dir, prefix), best_output_dir)
        logger.info("Saving model checkpoint to %s", best_output_dir)
        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
        save_labels(os.path.join(best_output_dir, "label.txt"), label_list)

    # evaluate
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("++++++++++++Validation+++++++++++++")
        log_fp.write("++++++++++++Validation+++++++++++++\n")
        global_step = max_metric_model_info["global_step"]
        logger.info("max %s global step: %d" % (args.max_metric_type, global_step))
        log_fp.write("max %s global step: %d\n" % (args.max_metric_type, global_step))
        prefix = "checkpoint-{}".format(global_step)
        checkpoint = os.path.join(args.output_dir, prefix)
        logger.info("checkpoint path: %s" % checkpoint)
        log_fp.write("checkpoint path: %s\n" % checkpoint)
        model = torch.load(os.path.join(checkpoint, "dnn.pkl"))
        model.to(args.device)
        model.eval()

        result = evaluate(args, model, dev_dataset, prefix=prefix, log_fp=log_fp)
        result = dict(("evaluation_" + k + "_{}".format(global_step), v) for k, v in result.items())
        logger.info(json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Testing
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("++++++++++++Testing+++++++++++++")
        log_fp.write("++++++++++++Testing+++++++++++++\n")
        global_step = max_metric_model_info["global_step"]
        logger.info("max %s global step: %d" % (args.max_metric_type, global_step))
        log_fp.write("max %s global step: %d\n" % (args.max_metric_type, global_step))
        prefix = "checkpoint-{}".format(global_step)
        checkpoint = os.path.join(args.output_dir, prefix)
        logger.info("checkpoint path: %s" % checkpoint)
        log_fp.write("checkpoint path: %s\n" % checkpoint)
        model = torch.load(os.path.join(checkpoint, "dnn.pkl"))
        model.to(args.device)
        model.eval()
        pred, true, result = predict(args, model, test_dataset, prefix=prefix, log_fp=log_fp)
        result = dict(("evaluation_" + k + "_{}".format(global_step), v) for k, v in result.items())
        logger.info(json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    log_fp.close()


def trainer(args, model, train_dataset, dev_dataset, test_dataset, log_fp=None):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
    if log_fp is None:
        log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    train_dataset_total_num = len(train_dataset)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    train_batch_total_num = len(train_dataloader)
    print("Train dataset len: %d, batch num: %d" % (train_dataset_total_num, train_batch_total_num))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_batch_total_num // args.gradient_accumulation_steps) + 1
    else:
        t_total = train_batch_total_num // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)

    # Train
    log_fp.write("***** Running training *****\n")
    logger.info("***** Running training *****")
    log_fp.write("Train Dataset Num examples = %d\n" % train_dataset_total_num)
    logger.info("Train Dataset  Num examples = %d", train_dataset_total_num)
    log_fp.write("Train Dataset Num Epochs = %d\n" % args.num_train_epochs)
    logger.info("Train Dataset Num Epochs = %d", args.num_train_epochs)
    log_fp.write("Train Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_train_batch_size)
    logger.info("Train Dataset Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    log_fp.write("Train Dataset Total train batch size (w. parallel, distributed & accumulation) = %d\n" % (args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
    logger.info("Train Dataset Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    log_fp.write("Train Dataset Gradient Accumulation steps = %d\n" % args.gradient_accumulation_steps)
    logger.info("Train Dataset Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    log_fp.write("Train Dataset Total optimization steps = %d\n" % t_total)
    logger.info("Train Dataset Total optimization steps = %d", t_total)
    log_fp.write("#" * 50 + "\n")
    log_fp.flush()

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    max_metric_type = args.max_metric_type
    max_metric_value = 0
    max_metric_model_info = {}
    last_max_metric_global_step = None
    cur_max_metric_global_step = None
    use_time = 0
    run_begin_time = time.time()
    real_epoch = 0

    for epoch in train_iterator:
        if args.tfrecords:
            epoch_iterator = tqdm(train_dataloader, total=train_batch_total_num, desc="Iteration", disable=args.local_rank not in [-1, 0])
        else:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            begin_time = time.time()
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "inputs": batch[0],
                "labels": batch[-1]
            }
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                # The loss of each batch is divided by gradient_accumulation_steps
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            epoch_iterator.set_description("loss {}".format(round(loss.item(), 5)))

            tr_loss += loss.item()
            end_time = time.time()
            use_time += (end_time - begin_time)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clear the gradient after completing gradient_accumulation_steps steps
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # evaluate per logging_steps steps
                update_flag = False
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        result = evaluate(args, model, dev_dataset, prefix="checkpoint-{}".format(global_step), log_fp=log_fp)
                        # update_flag = False
                        for key, value in result.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                            if key == max_metric_type:
                                if max_metric_value < value:
                                    max_metric_value = value
                                    update_flag = True
                                    last_max_metric_global_step = cur_max_metric_global_step
                                    cur_max_metric_global_step = global_step
                        logs["update_flag"] = update_flag
                        if update_flag:
                            max_metric_model_info.update({"epoch": epoch + 1, "global_step": global_step})
                            max_metric_model_info.update(logs)
                        _, _, test_result = predict(args, model, test_dataset, "checkpoint-{}".format(global_step), log_fp=log_fp)
                        for key, value in test_result.items():
                            eval_key = "test_{}".format(key)
                            logs[eval_key] = value
                    avg_iter_time = round(use_time / (args.gradient_accumulation_steps * args.logging_steps), 2)
                    logger.info("avg time per batch(s): %f\n" % avg_iter_time)
                    log_fp.write("avg time per batch (s): %f\n" % avg_iter_time)
                    use_time = 0
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logs["epoch"] = epoch + 1
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        if isinstance(value, dict):
                            for key1, value1 in value.items():
                                tb_writer.add_scalar(key + "_" + key1, value1, global_step)
                        else:
                            tb_writer.add_scalar(key, value, global_step)

                    logger.info(json.dumps({**logs, **{"step": global_step}}, ensure_ascii=False))
                    log_fp.write(json.dumps({**logs, **{"step": global_step}}, ensure_ascii=False) + "\n")
                    log_fp.write("##############################\n")
                    log_fp.flush()
                # save checkpoint per save_steps steps
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    print("save dir: ", output_dir)
                    if args.save_all:
                        save_check_point(args, model, output_dir)
                    elif update_flag:
                        if args.delete_old:
                            # delete the old chechpoint
                            filename_list = os.listdir(args.output_dir)
                            for filename in filename_list:
                                if "checkpoint-" in filename and filename != "checkpoint-{}".format(global_step):
                                    shutil.rmtree(os.path.join(args.output_dir, filename))
                        save_check_point(args, model, output_dir)
            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        real_epoch = epoch + 1
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    run_end_time = time.time()
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    log_fp.write(json.dumps(max_metric_model_info, ensure_ascii=False) + "\n")
    log_fp.write("##############################\n")
    avg_time_per_epoch = round((run_end_time - run_begin_time)/real_epoch, 2)
    logger.info("Avg time per epoch(s, %d epoch): %f\n" %(real_epoch, avg_time_per_epoch))
    log_fp.write("Avg time per epoch(s, %d epoch): %f\n" %(real_epoch, avg_time_per_epoch))

    return global_step, tr_loss / global_step, max_metric_model_info


def save_check_point(args, model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    torch.save(model_to_save, os.path.join(output_dir, "dnn.pkl"))
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, dataset, prefix, log_fp=None):
    '''
    evaluate
    :param args:
    :param model:
    :param dataset:
    :param prefix:
    :param log_fp:
    :return:
    '''
    output_dir = os.path.join(args.output_dir, prefix)
    print("Evaluating dir path: ", output_dir)
    if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_dir)
    result = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataset_total_num = len(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_batch_total_num = len(eval_dataloader)
    print("Dev dataset len: %d, batch num: %d" % (eval_dataset_total_num, eval_batch_total_num))

    # multi GPU
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # evaluate
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("Num examples = %d", eval_dataset_total_num)
    logger.info("Batch size = %d", args.eval_batch_size)
    if log_fp:
        log_fp.write("***** Running evaluation {} *****\n".format(prefix))
        log_fp.write("Dev Dataset Num examples = %d\n" % eval_dataset_total_num)
        log_fp.write("Dev Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_eval_batch_size)
        log_fp.write("Dev Dataset batch number = %d\n" % eval_batch_total_num)
        log_fp.write("#" * 50 + "\n")
    eval_loss = 0.0
    nb_eval_steps = 0
    # predicted prob
    pred_scores = None
    # ground truth
    out_label_ids = None

    for batch in tqdm(eval_dataloader, total=eval_batch_total_num, desc="Evaluating"):
        # evaluate
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "inputs": batch[0],
                "labels": batch[-1]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits, output = outputs[:3]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if pred_scores is None:
            pred_scores = output.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            pred_scores = np.append(pred_scores, output.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode in ["multi-label", "multi_label"]:
        result = metrics_multi_label(out_label_ids, pred_scores, threshold=0.5)
    elif args.output_mode in ["multi-class", "multi_class"]:
        result = metrics_multi_class(out_label_ids, pred_scores)
    elif args.output_mode == "regression":
        pass # to do
    elif args.output_mode in ["binary-class", "binary_class"]:
        result = metrics_binary(out_label_ids, pred_scores, threshold=0.5,
                                savepath=os.path.join(output_dir, "dev_confusion_matrix.png"))

    with open(os.path.join(output_dir, "dev_metrics.txt"), "w") as writer:
        logger.info("***** Eval Dev results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    logger.info("Dev metrics: ")
    logger.info(json.dumps(result, ensure_ascii=False))
    logger.info("")

    return result


def predict(args, model, dataset, prefix, log_fp=None):
    '''
    prediction
    :param args:
    :param model:
    :param dataset:
    :param prefix:
    :param log_fp:
    :return:
    '''
    output_dir = os.path.join(args.output_dir, prefix)
    print("Testing info save dir: ", output_dir)
    if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_dir)

    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    test_dataset_total_num = len(dataset)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.test_batch_size)
    test_batch_total_num = len(test_dataloader)
    print("Test dataset len: %d, batch len: %d" % (test_dataset_total_num, test_batch_total_num))

    # Multi GPU
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running test {} *****".format(prefix))
    logger.info("Num examples = %d", test_dataset_total_num)
    logger.info("Batch size = %d", args.test_batch_size)
    if log_fp:
        log_fp.write("***** Running testing {} *****\n".format(prefix))
        log_fp.write("Test Dataset Num examples = %d\n" % test_dataset_total_num)
        log_fp.write("Test Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_eval_batch_size)
        log_fp.write("Test Dataset batch number = %d\n" % test_batch_total_num)
        log_fp.write("#" * 50 + "\n")
    test_loss = 0.0
    nb_test_steps = 0
    # predicted prob
    pred_scores = None
    # ground truth
    out_label_ids = None
    for batch in tqdm(test_dataloader, total=test_batch_total_num, desc="Testing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "inputs": batch[0],
                "labels": batch[-1]
            }
            outputs = model(**inputs)
            tmp_test_loss, logits, output = outputs[:3]

            test_loss += tmp_test_loss.mean().item()
        nb_test_steps += 1
        if pred_scores is None:
            pred_scores = output.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            pred_scores = np.append(pred_scores, output.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    test_loss = test_loss / nb_test_steps
    if args.output_mode in ["multi_class", "multi-class"]:
        label_list = load_labels(filepath=args.label_filepath, header=True)
        pred_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=pred_scores, threshold=0.5)
        true_label_names = [label_list[idx] for idx in out_label_ids]
    elif args.output_mode == "regression":
        preds = np.squeeze(pred_scores)
        pred_label_names = list(preds)
        true_label_names = list(out_label_ids)
    elif args.output_mode in ["multi_label", "multi-label"]:
        label_list = load_labels(filepath=args.label_filepath, header=True)
        pred_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=pred_scores, threshold=0.5)
        true_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=out_label_ids, threshold=0.5)
    elif args.output_mode in ["binary_class", "binary-class"]:
        label_list = load_labels(filepath=args.label_filepath, header=True)
        pred_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=pred_scores, threshold=0.5)
        true_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=out_label_ids, threshold=0.5)

    if args.output_mode in ["multi_class", "multi-class"]:
        result = metrics_multi_class(out_label_ids, pred_scores)
    elif args.output_mode in ["multi_label", "multi-label"]:
        result = metrics_multi_label(out_label_ids, pred_scores, threshold=0.5)
    elif args.output_mode == "regression":
        pass # to do
    elif args.output_mode in ["binary_class", "binary-class"]:
        result = metrics_binary(out_label_ids, pred_scores, threshold=0.5,
                                savepath=os.path.join(output_dir, "test_confusion_matrix.png"))

    with open(os.path.join(output_dir, "test_results.txt"), "w") as wfp:
        for idx in range(len(pred_label_names)):
            wfp.write("%d,%s,%s\n" %(idx, str(pred_label_names[idx]), str(true_label_names[idx])))

    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as wfp:
        logger.info("***** Eval Test results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            wfp.write("%s = %s\n" % (key, str(result[key])))

    logger.info("Test metrics: ")
    logger.info(json.dumps(result, ensure_ascii=False))
    logger.info("")

    return pred_label_names, true_label_names, result


if __name__ == "__main__":
    run()