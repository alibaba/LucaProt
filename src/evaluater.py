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
@datetime: 2022/11/28 19:18
@project: DeepProtFunc
@file: evaluater
@desc: evaluate validation in model building
'''
import logging, json
import pandas as pd
import numpy as np
import os, torch, sys
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler
from data_loader import load_and_cache_examples, load_and_cache_examples_for_tfrecords
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../src/common")
try:
    from multi_label_metrics import metrics_multi_label
    from metrics import metrics_multi_class, metrics_binary
except ImportError:
    from src.common.multi_label_metrics import metrics_multi_label
    from src.common.metrics import metrics_multi_class, metrics_binary

logger = logging.getLogger(__name__)


def evaluate(args, model, processor, seq_tokenizer, subword, struct_tokenizer, prefix="", log_fp=None):
    '''
    evaluation
    :param args:
    :param model:
    :param processor:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param prefix:
    :param log_fp:
    :return:
    '''
    output_dir = os.path.join(args.output_dir, prefix)
    print("Evaluating information dir: ", output_dir)
    if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_dir)
    result = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.tfrecords:
        eval_dataset, eval_dataset_total_num = load_and_cache_examples_for_tfrecords(args, processor, seq_tokenizer, subword, struct_tokenizer, evaluate=True, predict=False, log_fp=log_fp)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.train_batch_size)
        eval_batch_total_num = (eval_dataset_total_num + args.eval_batch_size - 1) // args.eval_batch_size
    else:
        eval_dataset = load_and_cache_examples(args, processor, seq_tokenizer, subword, struct_tokenizer, evaluate=True, predict=False, log_fp=log_fp)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataset_total_num = len(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_batch_total_num = len(eval_dataloader)
    print("Dev dataset len: %d, batch num: %d" % (eval_dataset_total_num, eval_batch_total_num))

    # Multi GPU
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
            if args.model_type == "sequence":
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[-1]
                }
            elif args.model_type == "embedding":
                inputs = {
                    "embedding_info": batch[0],
                    "embedding_attention_mask": batch[1] if args.embedding_type != "bos" else None,
                    "labels": batch[-1]
                }
            elif args.model_type == "structure":
                inputs = {
                    "struct_input_ids": batch[0],
                    "struct_contact_map": batch[1],
                    "labels": batch[-1]
                }
            elif args.model_type == "sefn":
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "embedding_info": batch[4],
                    "embedding_attention_mask": batch[5] if args.embedding_type != "bos" else None,
                    "labels": batch[-1],
                }
            elif args.model_type == "ssfn":
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "struct_input_ids": batch[4],
                    "struct_contact_map": batch[5],
                    "labels": batch[-1],
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
