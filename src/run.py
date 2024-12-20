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
@datetime: 2022/11/26 21:02
@project: DeepProtFunc
@file: run
@desc: model building
'''
import os
import sys
import json
import logging
import codecs
import argparse
import shutil
from subword_nmt.apply_bpe import BPE
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src/SSFN")
sys.path.append("../src/common")
try:
    from common.metrics import metrics_multi_class, metrics_binary
    from common.multi_label_metrics import *
    from utils import set_seed, save_labels, get_parameter_number, load_trained_model
    from trainer import train
    from evaluater import evaluate
    from predictor import predict
    from data_loader import SequenceStructureProcessor
    from SSFN.model import *
except ImportError:
    from src.common.metrics import metrics_multi_class, metrics_binary
    from src.common.multi_label_metrics import *
    from src.utils import set_seed, save_labels, get_parameter_number, load_trained_model
    from src.trainer import train
    from src.evaluater import evaluate
    from src.predictor import predict
    from src.data_loader import SequenceStructureProcessor
    from src.SSFN.model import *


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser("Model Building for LucaProt")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="the dataset dirpath."
    )
    parser.add_argument(
        "--separate_file",
        action="store_true",
        help="load the entire dataset using memory, only the names of the pdb and embedding files are listed in the train/dev/test.csv")
    parser.add_argument(
        "--filename_pattern",
        default=None,
        type=str,
        help="the dataset filename pattern，such as {}_with_pdb_emb.csv including train_with_pdb_emb.csv, dev_with_pdb_emb.csv, test_with_pdb_emb.csv in ${data_dir}")
    parser.add_argument(
        "--tfrecords",
        action="store_true",
        help="whether the dataset is in the tfrecords. When true, only the specified number of samples(${shuffle_queue_size}) are loaded into memory at once, the tfrecords must  be in ${data_dir}/tfrecords/train/xxx.tfrecords, ${data_dir}/tfrecords/dev/xxx.tfrecords and ${data_dir}/tfrecords/test/xxx.tfrecords. xxx.tfrecords may be 01-of-01.tfrecords(only including sequence)、01-of-01_emb.records(including sequence and structural embedding)、01-of-01_pdb_emb.records(including sequence, 3d-structure contact map, and structural embedding)")
    parser.add_argument(
        "--shuffle_queue_size",
        default=5000,
        type=int,
        help="how many samples are loaded into memory at once"
    )
    parser.add_argument(
        "--multi_tfrecords",
        action="store_true",
        help="whether exists multi-tfrecords"
    )
    parser.add_argument(
        "--dataset_name",
        default="rdrp_40_extend",
        type=str,
        required=True,
        help="dataset name"
    )
    parser.add_argument(
        "--dataset_type",
        default="protein",
        type=str,
        required=True,
        choices=["protein", "dna", "rna"],
        help="dataset type"
    )
    parser.add_argument(
        "--task_type",
        default="binary_class",
        type=str,
        required=True,
        choices=["multi_label", "multi_class", "binary_class"],
        help="task type"
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        choices=["sequence", "structure", "embedding", "sefn", "ssfn"],
        help="model type selected in the list: ['sequence-based', 'structure-based', 'structural embedding based', 'sequence and structural embedding based', 'sequence and structure based']"
    )
    parser.add_argument(
        "--subword",
        action="store_true",
        help="whether use subword-level for sequence"
    )
    parser.add_argument(
        "--codes_file",
        type=str,
        default="../subword/rdrp/protein_codes_rdrp_20000.txt",
        help="subword codes filepath"
    )

    parser.add_argument(
        "--input_mode",
        type=str,
        default="single",
        choices=["single", "concat", "independent"],
        help="the input operation"
    )
    parser.add_argument(
        "--label_type",
        default="rdrp",
        type=str,
        required=True,
        help="label type"
    )
    parser.add_argument(
        "--label_filepath",
        default=None,
        type=str,
        required=True,
        help="the label list filepath"
    )

    # for structure
    parser.add_argument(
        "--cmap_type",
        default=None,
        type=str,
        choices=["C_alpha", "C_bert"],
        help="the calculation type of 3d-structure contact map"
    )
    parser.add_argument(
        "--cmap_thresh",
        default=10.0,
        type=float,
        help="contact map threshold."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="the output dirpath"
    )

    parser.add_argument(
        "--log_dir",
        default="./logs/",
        type=str,
        required=True,
        help="log dir."
    )
    parser.add_argument(
        "--tb_log_dir",
        default="./tb-logs/",
        type=str,
        required=True,
        help="tensorboard log dir."
    )

    # Other parameters
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        required=True,
        help="the config filepath of the running model"
    )
    parser.add_argument(
        "--seq_vocab_path",
        default=None,
        type=str,
        help="sequence token vocab filepath"
    )
    parser.add_argument(
        "--struct_vocab_path",
        default=None,
        type=str,
        help="structure node token vocab filepath"
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="cache dirpath"
    )

    # sequence pooling_type
    parser.add_argument(
        "--seq_pooling_type",
        type=str,
        default=None,
        choices=["none", "sum", "max", "avg", "attention", "context_attention", "weighted_attention", "value_attention", "transformer"],
        help="pooling type for sequence encoder"
    )
    # structure pooling_type
    parser.add_argument(
        "--struct_pooling_type",
        type=str,
        default=None,
        choices=["sum", "max", "avg", "attention", "context_attention", "weighted_attention", "value_attention", "transformer"],
        help="pooling type for structure encoder"
    )
    # embedding pooling_type
    parser.add_argument(
        "--embedding_pooling_type",
        type=str,
        default=None,
        choices=["none", "sum", "max", "avg", "attention", "context_attention", "weighted_attention", "value_attention", "transformer"],
        help="pooling type for embedding encoder"
    )
    # activate function
    parser.add_argument(
        "--activate_func",
        type=str,
        default=None,
        choices=["tanh", "relu", "leakyrelu", "gelu"],
        help="activate function type after pooling"
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to run predict on the test set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=50,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
             "See details at https://nvidia.github.io/apex/amp.html"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )

    # multi-label/ binary-class
    parser.add_argument(
        "--sigmoid",
        action="store_true",
        help="classifier add sigmoid if task_type is binary-class or multi-label"
    )

    # loss func
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce",
        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
        help="loss type"
    )

    # which metric for model finalization selected
    parser.add_argument(
        "--max_metric_type",
        type=str,
        default="f1",
        required=True,
        choices=["acc", "jaccard", "prec", "recall", "f1", "fmax", "roc_auc", "pr_auc"],
        help="which metric for model selected"
    )

    # for BCE Loss
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=40,
        help="positive weight for bce"
    )

    # for CE Loss
    parser.add_argument(
        "--weight",
        type=str,
        default=None,
        help="positive weight for multi-class"
    )

    # for focal Loss
    parser.add_argument(
        "--focal_loss_alpha",
        type=float,
        default=0.7,
        help="focal loss alpha"
    )
    parser.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=2.0,
        help="focal loss gamma"
    )
    parser.add_argument(
        "--focal_loss_reduce",
        action="store_true",
        help="mean for one sample(default sum)"
    )

    # for asymmetric Loss
    parser.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=4.0,
        help="negative gamma for asl"
    )
    parser.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=1.0,
        help="positive gamma for asl"
    )

    # for sequence and structure graph node size(contact map shape)
    parser.add_argument(
        "--seq_max_length",
        default=2048,
        type=int,
        help="the length of input sequence more than max length will be truncated, shorter will be padded."
    )
    parser.add_argument(
        "--struct_max_length",
        default=2048,
        type=int,
        help="the length of input contact map more than max length will be truncated, shorter will be padded."
    )
    parser.add_argument(
        "--trunc_type",
        default="right",
        type=str,
        required=True,
        choices=["left", "right"],
        help="truncate type for whole input"
    )
    parser.add_argument(
        "--no_position_embeddings",
        action="store_true",
        help="Whether not to use position_embeddings"
    )
    parser.add_argument(
        "--no_token_type_embeddings",
        action="store_true",
        help="Whether not to use token_type_embeddings"
    )

    # for embedding input
    parser.add_argument(
        "--embedding_input_size",
        default=2560,
        type=int,
        help="the length of input embedding dim."
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="matrix",
        choices=[None, "contacts", "bos", "matrix"],
        help="the type of the structural embedding info"
    )
    parser.add_argument(
        "--embedding_max_length",
        default=2048,
        type=int,
        help="the length of input embedding more than max length will be truncated, shorter will be padded."
    )

    parser.add_argument(
        "--model_dirpath",
        default=None,
        type=str,
        help="load the trained model to continue training."
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="save all checkpoints during training"
    )
    parser.add_argument(
        "--delete_old",
        action="store_true",
        help="delete old checkpoint by the specific metric"
    )
    args = parser.parse_args()

    if args.model_type == "sequence":
        output_input_col_names = [args.dataset_type, "seq"]
        args.has_seq_encoder = True
        args.has_struct_encoder = False
        args.has_embedding_encoder = False
        args.cmap_type = None
        args.embedding_type = None
    elif args.model_type == "structure":
        output_input_col_names = [args.dataset_type, "structure"]
        args.has_seq_encoder = False
        args.has_struct_encoder = True
        args.has_embedding_encoder = False
        args.embedding_type = None
    elif args.model_type == "embedding":
        output_input_col_names = [args.dataset_type, "embedding"]
        args.has_seq_encoder = False
        args.has_struct_encoder = False
        args.has_embedding_encoder = True
        args.cmap_type = None
    elif args.model_type == "sefn":
        output_input_col_names = [args.dataset_type, "seq", "embedding"]
        args.has_seq_encoder = True
        args.has_struct_encoder = False
        args.has_embedding_encoder = True
        args.cmap_type = None
    elif args.model_type == "ssfn":
        output_input_col_names = [args.dataset_type, "seq", "structure"]
        args.has_seq_encoder = True
        args.has_struct_encoder = True
        args.has_embedding_encoder = False
        args.embedding_type = None
    else:
        raise Exception("Not support this model_type=%s" % args.model_type)

    # overwrite the output dir
    if os.path.exists(args.output_dir) \
            and os.listdir(args.output_dir) \
            and args.do_train \
            and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)
        )
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    # create the logs dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    # create tensorboard logs dir
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # input types
    log_fp.write("Inputs:\n")
    log_fp.write("Input Name List: %s\n" % ",".join(output_input_col_names))
    log_fp.write("#" * 50 + "\n")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1 if args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("#" * 50)
    logger.info(str(args))
    logger.info("#" * 50)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s | %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning(
        "Process rank: %d, device: %s, n_gpu: %d, distributed training: %r, 16-bits training: %r" % (
            args.local_rank,
            str(device),
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16
        )
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.dataset_name = args.dataset_name.lower()

    # Data processor, different processors for different tasks
    processor = SequenceStructureProcessor(
        model_type=args.model_type,
        separate_file=args.separate_file,
        filename_pattern=args.filename_pattern
    )

    # the output type
    args.output_mode = args.task_type
    # For binary_class/multi_label tasks, the sigmoid needs to be added to the last layer
    if args.output_mode in ["multi_label", "multi-label", "binary_class", "binary-class"]:
        args.sigmoid = True

    # get label list
    label_list = processor.get_labels(
        label_filepath=args.label_filepath
    )
    num_labels = len(label_list)
    logger.info("#" * 25 + "Labels Num:" + "#" * 25)
    logger.info("Num Labels: %d" % num_labels)
    save_labels(os.path.join(args.log_dir, "label.txt"), label_list)

    # Get different task models according to the task type name
    args.model_type = args.model_type.lower()
    # Load config
    config_class = BertConfig
    config = config_class(**json.load(open(args.config_path, "r")))
    config.max_position_embeddings = int(args.seq_max_length)
    config.num_labels = num_labels
    config.embedding_pooling_type = args.embedding_pooling_type
    if args.activate_func:
        config.activate_func = args.activate_func
    if args.pos_weight:
        config.pos_weight = args.pos_weight
    # tokenization
    subword = None
    if args.has_seq_encoder:
        seq_tokenizer_class = BertTokenizer
        seq_tokenizer = seq_tokenizer_class(args.seq_vocab_path, do_lower_case=args.do_lower_case)
        config.vocab_size = seq_tokenizer.vocab_size
        if args.subword:
            bpe_codes_prot = codecs.open(args.codes_file)
            subword = BPE(bpe_codes_prot, merges=-1, separator='')
    else:
        seq_tokenizer_class = None
        seq_tokenizer = None

    if args.has_struct_encoder:
        struct_tokenizer_class = BertTokenizer
        struct_tokenizer = struct_tokenizer_class(args.struct_vocab_path, do_lower_case=args.do_lower_case)
        config.struct_vocab_size = struct_tokenizer.vocab_size
    else:
        struct_tokenizer_class = None
        struct_tokenizer = None

    # model class
    model_class = SequenceAndStructureFusionNetwork
    if args.model_dirpath and os.path.exists(args.model_dirpath):
        model = load_trained_model(config, args, model_class, args.model_dirpath)
    else:
        model = model_class(config, args)

    # output model hyperparameters in logger
    if len(config.id2label) > 10:
        str_config = copy.deepcopy(config)
        str_config.id2label = {}
        str_config.label2id = {}
    else:
        str_config = copy.deepcopy(config)
    log_fp.write("Model Config:\n %s\n" % str(str_config))
    log_fp.write("#" * 50 + "\n")
    log_fp.write("Mode Architecture:\n %s\n" % str(model))
    log_fp.write("#" * 50 + "\n")

    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # output training/evaluation hyperparameters in logger
    logger.info("====Training/Evaluation Parameters:=====")
    for attr, value in sorted(args.__dict__.items()):
        logger.info("\t{}={}".format(attr, value))
    logger.info("====Parameters End=====\n")

    args_dict = {}
    for attr, value in sorted(args.__dict__.items()):
        if attr != "device":
            args_dict[attr] = value

    log_fp.write(json.dumps(args_dict, ensure_ascii=False) + "\n")
    log_fp.write("#" * 50 + "\n")
    log_fp.write("num labels: %d\n" % num_labels)
    log_fp.write("#" * 50 + "\n")

    model_size_info = get_parameter_number(model)
    log_fp.write(json.dumps(model_size_info, ensure_ascii=False) + "\n")
    log_fp.write("#" * 50 + "\n")
    log_fp.flush()

    # Training
    max_metric_model_info = None
    if args.do_train:
        logger.info("++++++++++++Training+++++++++++++")
        global_step, tr_loss, max_metric_model_info = train(
            args,
            model,
            processor,
            seq_tokenizer,
            subword,
            struct_tokenizer=struct_tokenizer,
            log_fp=log_fp
        )
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Save
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("++++++++++++Save Model+++++++++++++")
        # Create output directory if needed
        best_output_dir = os.path.join(args.output_dir, "best")
        global_step = max_metric_model_info["global_step"]
        prefix = "checkpoint-{}".format(global_step)
        shutil.copytree(os.path.join(args.output_dir, prefix), best_output_dir)
        logger.info("Saving model checkpoint to %s", best_output_dir)
        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
        save_labels(os.path.join(best_output_dir, "label.txt"), label_list)

    # Evaluate
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("++++++++++++Validation+++++++++++++")
        log_fp.write("++++++++++++Validation+++++++++++++\n")
        global_step = max_metric_model_info["global_step"]
        logger.info("max %s global step: %d" % (args.max_metric_type, global_step))
        log_fp.write("max %s global step: %d\n" % (args.max_metric_type, global_step))
        prefix = "checkpoint-{}".format(global_step)
        checkpoint = os.path.join(args.output_dir, prefix)
        if seq_tokenizer is None and seq_tokenizer_class:
            seq_tokenizer = seq_tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
        if struct_tokenizer_class and struct_tokenizer is None:
            struct_tokenizer = struct_tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)

        logger.info("checkpoint path: %s" % checkpoint)
        log_fp.write("checkpoint path: %s\n" % checkpoint)
        model = model_class.from_pretrained(checkpoint, args=args)
        model.to(args.device)
        result = evaluate(args, model, processor, seq_tokenizer, subword, struct_tokenizer, prefix=prefix, log_fp=log_fp)
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
        if seq_tokenizer is None and seq_tokenizer_class:
            seq_tokenizer = seq_tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
        if struct_tokenizer_class and struct_tokenizer is None:
            struct_tokenizer = struct_tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
        logger.info("checkpoint path: %s" % checkpoint)
        log_fp.write("checkpoint path: %s\n" % checkpoint)
        model = model_class.from_pretrained(checkpoint, args=args)
        model.to(args.device)
        pred, true, result = predict(args, model, processor, seq_tokenizer, subword, struct_tokenizer, prefix=prefix, log_fp=log_fp)
        result = dict(("evaluation_" + k + "_{}".format(global_step), v) for k, v in result.items())
        logger.info(json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    # close fp
    if args.local_rank in [-1, 0] and log_fp:
        log_fp.close()

    torch.distributed.barrier()


if __name__ == "__main__":
    main()


