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
@datetime: 2022/11/28 19:13
@project: DeepProtFunc
@file: trainer
@desc: trainer on training dataset for model building
'''
import os
import sys
import json
import logging
import torch
import time
import shutil
from utils import set_seed
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from data_loader import load_and_cache_examples, load_and_cache_examples_for_tfrecords
    from evaluater import evaluate
    from predictor import predict
except ImportError:
    from src.data_loader import load_and_cache_examples, load_and_cache_examples_for_tfrecords
    from src.evaluater import evaluate
    from src.predictor import predict

logger = logging.getLogger(__name__)


def train(
        args,
        model,
        processor,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        log_fp=None
):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
    if log_fp is None:
        log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.tfrecords:
        train_dataset, train_dataset_total_num = load_and_cache_examples_for_tfrecords(
            args,
            processor,
            seq_tokenizer,
            subword,
            struct_tokenizer,
            evaluate=False,
            predict=False,
            log_fp=log_fp
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size
        )
        train_batch_total_num = (train_dataset_total_num + args.train_batch_size - 1) // args.train_batch_size
    else:
        train_dataset = load_and_cache_examples(
            args,
            processor,
            seq_tokenizer,
            subword,
            struct_tokenizer,
            evaluate=False,
            predict=False,
            log_fp=log_fp
        )
        train_dataset_total_num = len(train_dataset)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size
        )
        train_batch_total_num = len(train_dataloader)
    print("Train dataset len: %d, batch num: %d" % (
        train_dataset_total_num,
        train_batch_total_num
    ))

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
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
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
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # Train
    log_fp.write("***** Running training *****\n")
    logger.info("***** Running training *****")
    log_fp.write("Train Dataset Num examples = %d\n" % train_dataset_total_num)
    logger.info("Train Dataset  Num examples = %d" % train_dataset_total_num)
    log_fp.write("Train Dataset Num Epochs = %d\n" % args.num_train_epochs)
    logger.info("Train Dataset Num Epochs = %d" % args.num_train_epochs)
    log_fp.write("Train Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_train_batch_size)
    logger.info("Train Dataset Instantaneous batch size per GPU = %d" % args.per_gpu_train_batch_size)
    log_fp.write("Train Dataset Total train batch size (w. parallel, distributed & accumulation) = %d\n" % (
            args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    ))
    logger.info("Train Dataset Total train batch size (w. parallel, distributed & accumulation) = %d" % (
            args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    ))
    log_fp.write("Train Dataset Gradient Accumulation steps = %d\n" % args.gradient_accumulation_steps)
    logger.info("Train Dataset Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)
    log_fp.write("Train Dataset Total optimization steps = %d\n" % t_total)
    logger.info("Train Dataset Total optimization steps = %d" % t_total)
    log_fp.write("#" * 50 + "\n")
    log_fp.flush()

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

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
            epoch_iterator = tqdm(
                train_dataloader,
                total=train_batch_total_num,
                desc="Iteration",
                disable=args.local_rank not in [-1, 0]
            )
        else:
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=args.local_rank not in [-1, 0]
            )
        for step, batch in enumerate(epoch_iterator):
            begin_time = time.time()
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
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
            else:
                # to do
                inputs = {}
            outputs = model(
                **inputs
            )
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                # The loss of each batch will be divided by gradient_accumulation_steps
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
                # Clear the gradient after completing the gradient_accumulation_steps steps
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # evaluate per logging_steps
                update_flag = False
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        result = evaluate(
                            args,
                            model,
                            processor,
                            seq_tokenizer,
                            subword,
                            struct_tokenizer,
                            prefix="checkpoint-{}".format(global_step),
                            log_fp=log_fp
                        )
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
                        _, _, test_result = predict(
                            args,
                            model,
                            processor,
                            seq_tokenizer,
                            subword,
                            struct_tokenizer,
                            "checkpoint-{}".format(global_step), log_fp=log_fp
                        )
                        for key, value in test_result.items():
                            eval_key = "test_{}".format(key)
                            logs[eval_key] = value
                    avg_iter_time = round(
                        use_time/(args.gradient_accumulation_steps * args.logging_steps),
                        2
                    )
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
                # save checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    print("save dir: ", output_dir)
                    if args.save_all:
                        save_check_point(args, model, seq_tokenizer, struct_tokenizer, output_dir)
                    elif update_flag:
                        if args.delete_old:
                            # delete the old chechpoint
                            filename_list = os.listdir(args.output_dir)
                            for filename in filename_list:
                                if "checkpoint-" in filename and filename != "checkpoint-{}".format(global_step):
                                    shutil.rmtree(os.path.join(args.output_dir, filename))
                        '''
                        if last_max_metric_global_step:
                            print("remove dir: ", os.path.join(args.output_dir, "checkpoint-{}".format(last_max_metric_global_step)))
                            shutil.rmtree(os.path.join(args.output_dir, "checkpoint-{}".format(last_max_metric_global_step)))
                        '''
                        save_check_point(args, model, seq_tokenizer, struct_tokenizer, output_dir)

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
    logger.info("Avg time per epoch(s, %d epoch): %f\n" % (real_epoch, avg_time_per_epoch))
    log_fp.write("Avg time per epoch(s, %d epoch): %f\n" % (real_epoch, avg_time_per_epoch))

    return global_step, tr_loss / global_step, max_metric_model_info


def save_check_point(args, model, seq_tokenizer, struct_tokenizer, output_dir):
    '''
    save checkpoint
    :param args:
    :param model:
    :param seq_tokenizer:
    :param struct_tokenizer:
    :param output_dir:
    :return:
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    if seq_tokenizer:
        seq_tokenizer_dir = os.path.join(output_dir, "sequence")
        if not os.path.exists(seq_tokenizer_dir):
            os.makedirs(seq_tokenizer_dir)
        seq_tokenizer.save_pretrained(seq_tokenizer_dir)
    if struct_tokenizer:
        struct_tokenizer_dir = os.path.join(output_dir, "structure")
        if not os.path.exists(struct_tokenizer_dir):
            os.makedirs(struct_tokenizer_dir)
        struct_tokenizer.save_pretrained(struct_tokenizer_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)
