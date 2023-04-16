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
@datetime: 2022/11/28 19:19
@project: DeepProtFunc
@file: predicter
@desc: evaluate tesing in model building
'''
import logging, json
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
    from utils import label_id_2_label_name
except ImportError:
    from src.common.multi_label_metrics import metrics_multi_label
    from src.common.metrics import metrics_multi_class, metrics_binary
    from src.utils import label_id_2_label_name

logger = logging.getLogger(__name__)


def predict(args, model, processor, seq_tokenizer, subword, struct_tokenizer, prefix="", log_fp=None):
    '''
    prediction during model building
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
    print("Testing info save dir: ", output_dir)
    if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_dir)

    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.tfrecords:
        test_dataset, test_dataset_total_num = load_and_cache_examples_for_tfrecords(args, processor, seq_tokenizer, subword, struct_tokenizer, evaluate=False, predict=True, log_fp=log_fp)
        test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size)
        test_batch_total_num = (test_dataset_total_num + args.test_batch_size - 1) // args.test_batch_size
    else:
        test_dataset = load_and_cache_examples(args, processor, seq_tokenizer, subword, struct_tokenizer, evaluate=False, predict=True, log_fp=log_fp)
        # Note that DistributedSampler samples randomly
        test_dataset_total_num = len(test_dataset)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)
        test_batch_total_num = len(test_dataloader)
    print("Test dataset len: %d, batch len: %d" % (test_dataset_total_num, test_batch_total_num))

    # multi gpu
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
    # prediction prob
    pred_scores = None
    # ground truth
    out_label_ids = None
    for batch in tqdm(test_dataloader, total=test_batch_total_num, desc="Testing"):
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
        label_list = processor.get_labels(label_filepath=args.label_filepath)
        pred_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=pred_scores, threshold=0.5)
        true_label_names = [label_list[idx] for idx in out_label_ids]
    elif args.output_mode == "regression":
        preds = np.squeeze(pred_scores)
        pred_label_names = list(preds)
        true_label_names = list(out_label_ids)
    elif args.output_mode in ["multi_label", "multi-label"]:
        label_list = processor.get_labels(label_filepath=args.label_filepath)
        pred_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=pred_scores, threshold=0.5)
        true_label_names = label_id_2_label_name(args.output_mode, label_list=label_list, prob=out_label_ids, threshold=0.5)
    elif args.output_mode in ["binary_class", "binary-class"]:
        label_list = processor.get_labels(label_filepath=args.label_filepath)
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

