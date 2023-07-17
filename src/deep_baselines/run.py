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
@datetime: 2023/3/28 15:48
@project: DeepProtFunc
@file: run.py
@desc: run deep baselines
'''

import os, sys
import argparse
import time, json
import shutil
import logging
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from deep_baselines.cheer import CatWCNN, WDCNN, WCNN, seq_encode as cheer_seq_encode
    from deep_baselines.virhunter import VirHunter, seq_encode as virhunter_seq_encode, one_hot_encode as virhunter_seq_hot_encode
    from deep_baselines.virtifier import Virtifier, seq_encode as virtifier_seq_encode
    from deep_baselines.virseeker import VirSeeker, seq_encode as virseeker_seq_encode
    from common.loss import AsymmetricLossOptimized, FocalLoss, MultiLabel_CCE
    from utils import *
    from multi_label_metrics import *
    from metrics import *
except ImportError:
    from src.deep_baselines.cheer import CatWCNN, WDCNN, WCNN, seq_encode as cheer_seq_encode
    from src.deep_baselines.virhunter import VirHunter, seq_encode as virhunter_seq_encode, one_hot_encode as virhunter_seq_hot_encode
    from src.deep_baselines.virtifier import Virtifier, seq_encode as virtifier_seq_encode
    from src.deep_baselines.virseeker import VirSeeker, seq_encode as virseeker_seq_encode
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
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=["CHEER-CatWCNN", "CHEER-WDCNN", "CHEER-WCNN", "VirHunter", "Virtifier", "VirSeeker"],  help="model type.")
    parser.add_argument("--label_type", default="rdrp", type=str, required=True, help="label type.")
    parser.add_argument("--label_filepath", default=None, type=str, required=True, help="label filepath.")
    parser.add_argument("--seq_vocab_path", default=None, type=str, help="sequence token vocab filepath")

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

    # for all
    parser.add_argument("--seq_max_length", default=2048, type=int, help="the length of input sequence more than max length will be truncated, shorter will be padded.")
    parser.add_argument("--trunc_type", default="right", type=str, required=True,  choices=["left", "right"], help="truncate type for whole input")
    parser.add_argument("--embedding_trainable", action="store_true", help="whether to train the embedding matrix")
    parser.add_argument("--embedding_dim", default=128, type=int, help="the dim of embedding vector")

    # for CHEER
    parser.add_argument("--channel_in", default=None, type=int, help="channel in")

    # for CHEER and VirHunter
    parser.add_argument("--kernel_nums", default=None, type=str, help="kernel_nums or kernel_num")
    parser.add_argument("--kernel_sizes", default=None, type=str, help="kernel_sizes or kernel_size")
    parser.add_argument("--fc_sizes", default=None, type=str,  help="fc_sizes or fc_size")

    # for VirHunter
    parser.add_argument("--one_hot_encode", action="store_true", help="use one hot encode")

    # for VirSeeker
    parser.add_argument("--embedding", action="store_true", help="using embedding")

    # for Virtifier
    parser.add_argument("--embedding_init", action="store_true", help="pre-trained embedding")
    parser.add_argument("--embedding_init_path", default=None, type=str, help="re-trained embedding filepath")
    parser.add_argument("--bidirectional", action="store_true", help="bidirectional of LSTM")
    parser.add_argument("--num_layers", default=1, type=int, help="num layers of LSTM")
    parser.add_argument("--hidden_dim", default=128, type=int, help="the dim of hidden vector")
    parser.add_argument("--padding_idx", default=0, type=int,  help="padding idx")

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
    if args.seq_max_length:
        config.max_position_embeddings = args.seq_max_length
    config.embedding_trainable = args.embedding_trainable
    if args.embedding_dim:
        config.embedding_dim = args.embedding_dim
    if args.model_type in ["CHEER-CatWCNN", "CHEER-WDCNN", "CHEER-WCNN"]:
        if args.embedding_dim:
            config.embedding_dim = args.embedding_dim
        if args.channel_in:
            config.channel_in = args.channel_in
        if args.kernel_nums:
            config.kernel_nums = list(args.kernel_nums.split(","))
        if args.kernel_sizes:
            config.kernel_sizes = list(args.kernel_sizes.split(","))
        if args.fc_sizes:
            fc_sizes = args.fc_sizes.split(",")
            if args.model_type == "CHEER-CatWCNN":
                config.fc_size1 = fc_sizes[0]
                config.fc_size2 = fc_sizes[1]
            else:
                config.fc_size = fc_sizes[0]
    elif args.model_type == "VirHunter":
        if args.kernel_nums:
            config.kernel_num = list(args.kernel_nums.split(","))[0]
        if args.kernel_sizes:
            config.kernel_size = list(args.kernel_sizes.split(","))[0]
        if args.fc_sizes:
            fc_sizes = args.fc_sizes.split(",")
            config.fc_size = fc_sizes[0]
        config.one_hot_encode = args.one_hot_encode
    elif args.model_type == "Virtifier":
        if args.embedding_init:
            config.embedding_init = args.embedding_init
        if args.embedding_init_path:
            config.embedding_init_path = args.embedding_init_path
        if args.bidirectional:
            config.bidirectional = args.bidirectional
        if args.num_layers:
            config.num_layers = args.num_layers
        if args.hidden_dim:
            config.hidden_dim = args.hidden_dim
        if args.padding_idx:
            config.padding_idx = args.padding_idx
        if args.fc_sizes:
            fc_sizes = args.fc_sizes.split(",")
            config.fc_size = fc_sizes[0]
    elif args.model_type == "VirSeeker":
        config.embedding = args.embedding
        config.bidirectional = args.bidirectional
        if args.num_layers:
            config.num_layers = args.num_layers
        if args.hidden_dim:
            config.hidden_dim = args.hidden_dim
        if args.padding_idx:
            config.padding_idx = args.padding_idx


def get_labels(label_filepath):
    '''
    get labels from file, exists header
    :param label_filepath:
    :return:
    '''
    with open(label_filepath, "r") as fp:
        labels = []
        multi_cols = False
        cnt = 0
        for line in fp.readlines():
            cnt += 1
            if cnt == 1:
                if line.find(",") > 0:
                    multi_cols = True
                continue
            line = line.strip()
            if multi_cols:
                idx = line.find(",")
                if idx > 0:
                    label_name = line[idx+1:].strip()
                else:
                    label_name = line
            else:
                label_name = line
            labels.append(label_name)
        return labels


def load_vocab(vocab_filepath):
    token_list = []
    with open(vocab_filepath, "r") as rfp:
        for line in rfp:
            token_list.append(line.strip())
    int_to_token = {idx: token for idx, token in enumerate(token_list)}
    token_to_int = {token: idx for idx, token in int_to_token.items()}
    return int_to_token, token_to_int


def load_dataset(args, dataset_type, encode_func, encode_func_args):
    '''
    load dataset
    :param args:
    :param dataset_type:
    :param encode_func: encode function
    :param encode_func_args: encode function args
    :return:
    '''
    x = []
    y = []
    lens = []
    if os.path.exists(args.label_filepath):
        label_list = load_labels(args.label_filepath, header=True)
    else:
        label_list = load_labels(os.path.join(args.data_dir, "label.txt"), header=True)
    label_map = {name: idx for idx, name in enumerate(label_list)}
    npz_filpath = os.path.join(args.data_dir, "%s_%s_%s.npz" % (dataset_type, args.model_type, str(args.one_hot_encode)))
    if os.path.exists(npz_filpath):
        npzfile = np.load(npz_filpath, allow_pickle=True)
        x = npzfile["x"]
        y = npzfile["y"]
        lens = npzfile["lens"]
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
            encode_func_args["seq"] = seq.upper()
            seq_ids, actural_len = encode_func(**encode_func_args)
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
            x.append(seq_ids)
            y.append(label)
            lens.append(actural_len)
            cnt += 1
            if cnt % 10000 == 0:
                print("done %d" % cnt)
        x = np.array(x)
        y = np.array(y)
        lens = np.array(lens)
        np.savez(npz_filpath, x=x, y=y, lens=lens)
    print("%s: x.shape: %s, y.shape: %s, lens.shape: %s" %(dataset_type, str(x.shape), str(y.shape), str(lens.shape)))

    return torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(lens, dtype=torch.long)), label_list


def run():
    args = main()
    logging.basicConfig(format="%(asctime)s-%(levelname)s-%(name)s | %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # overwrite the output dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
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

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1 if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    int_2_token, token_2_int = load_vocab(args.seq_vocab_path)
    config.vocab_size = len(token_2_int)

    label_list = get_labels(label_filepath=args.label_filepath)
    save_labels(os.path.join(args.log_dir, "label.txt"), label_list)
    args.num_labels = len(label_list)

    args2config(args, config)

    if args.model_type in ["CHEER-CatWCNN", "CHEER-WDCNN", "CHEER-WCNN"]:
        if args.channel_in:
            args.seq_max_length = (args.seq_max_length//args.channel_in) * args.channel_in
        else:
            args.seq_max_length = (args.seq_max_length//config.channel_in) * config.channel_in

    if args.model_type == "CHEER-CatWCNN":
        model = CatWCNN(config, args)
    elif args.model_type == "CHEER-WDCNN":
        model = WDCNN(config, args)
    elif args.model_type == "CHEER-WCNN":
        model = WCNN(config, args)
    elif args.model_type == "VirHunter":
        model = VirHunter(config, args)
    elif args.model_type == "Virtifier":
        model = Virtifier(config, args)
    elif args.model_type == "VirSeeker":
        model = VirSeeker(config, args)
    else:
        raise Exception("not support model type: %s" % args.model_type)

    encode_func = None
    encode_func_args = {"max_len": args.seq_max_length, "vocab": token_2_int, "trunc_type": args.trunc_type}
    if args.model_type in ["CHEER-CatWCNN", "CHEER-WDCNN",  "CHEER-WCNN"]:
        encode_func = cheer_seq_encode
        encode_func_args["channel_in"] = args.channel_in
    elif args.model_type == "VirHunter":
        encode_func = virhunter_seq_encode
    elif args.model_type == "Virtifier":
        encode_func = virtifier_seq_encode
    elif args.model_type == "VirSeeker":
        encode_func = virseeker_seq_encode
    args_dict = {}
    for attr, value in sorted(args.__dict__.items()):
        if attr != "device":
            args_dict[attr] = value

    log_fp.write(json.dumps(args_dict, ensure_ascii=False) + "\n")

    model.to(args.device)

    train_dataset, label_list = load_dataset(args, "train", encode_func, encode_func_args)

    dev_dataset, _ = load_dataset(args, "dev", encode_func, encode_func_args)

    test_dataset, _ = load_dataset(args, "test", encode_func, encode_func_args)

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
        model = torch.load(os.path.join(checkpoint, "%s.pkl" % args.model_type))
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
        model = torch.load(os.path.join(checkpoint, "%s.pkl" % args.model_type))
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
                "x": batch[0],
                "labels": batch[1],
                "lengths": batch[2]
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
    torch.save(model_to_save, os.path.join(output_dir, "%s.pkl" % args.model_type))
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
                "x": batch[0],
                "labels": batch[1],
                "lengths": batch[2]
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
                "x": batch[0],
                "labels": batch[1],
                "lengths": batch[2]
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
