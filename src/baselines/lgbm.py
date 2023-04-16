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
@datetime: 2022/12/29 19:35
@project: DeepProtFunc
@file: lgbm
@desc: LGBM (based protein structure embeddding (matrix or vector) for classification
'''
import json
import logging
import os, sys
import pickle5
import argparse
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
import pandas as pd
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
try:
    from utils import *
    from multi_label_metrics import *
    from metrics import *
except ImportError:
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="input dir, including: *.csv/*.txt.")
    parser.add_argument("--separate_file", action="store_true", help="the id of each sample in the dataset is separate from its details")
    parser.add_argument("--tfrecords", action="store_true", help="whether the dataset is in tfrecords")
    parser.add_argument("--filename_pattern", default=None, type=str, help="the dataset filename pattern, such as {}_with_pdb_emb.csv including train_with_pdb_emb.csv, dev_with_pdb_emb.csv, and test_with_pdb_emb.csv in ${data_dir}")

    parser.add_argument("--dataset_name", default="rdrp_40_extend", type=str, required=True, help="dataset name")
    parser.add_argument("--dataset_type", default="protein", type=str, required=True, choices=["protein", "dna", "rna"], help="dataset type")
    parser.add_argument("--task_type", default="multi_label", type=str, required=True, choices=["multi_label", "multi_class", "binary_class"], help="task type")
    parser.add_argument("--label_type", default=None, type=str, required=True, help="label type.")
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
    parser.add_argument("--num_train_epochs", default=3.0, type=int, help="total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="if > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", help="overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="for distributed training: local_rank")
    parser.add_argument("--unbalance", action="store_true", help="set unbalance.")
    parser.add_argument("--pos_weight", type=float, default=40, help="positive weight")

    # which metric for model finalization selected
    parser.add_argument("--max_metric_type",  type=str, default="f1", required=True, choices=["acc", "jaccard", "prec", "recall", "f1", "fmax", "pr_auc", "roc_auc"], help="which metric for model selected")

    parser.add_argument("--early_stopping_rounds", default=None, type=int, help="early stopping rounds.")

    parser.add_argument("--grid_search", action="store_true", help="grid search for params or not")

    args = parser.parse_args()
    args.output_mode = args.task_type
    return args


def load_dataset(args, dataset_type):
    '''
    load the dataset
    :param args:
    :param dataset_type
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
    return x, y, label_list


def acc(probs, labeled_data):
    higher_better = True
    name = "acc"
    targets = labeled_data.get_label()
    if targets.ndim == 1:
        if len(np.unique(targets)) == 2:
            func = binary_f1
        else:
            func = multi_class_f1
    elif targets.ndim == 2 and targets.shape[1] > 2:
        func = multi_label_f1
    elif targets.ndim == 2 and targets.shape[1] == 2:
        func = binary_f1
    else:
        func = multi_class_f1
    value = func(targets, probs, threshold=0.5)
    return name, value, higher_better


def f1(probs, labeled_data):
    higher_better = True
    name = "f1"
    targets = labeled_data.get_label()
    if targets.ndim == 1:
        if len(np.unique(targets)) == 2:
            func = binary_f1
        else:
            func = multi_class_f1
    elif targets.ndim == 2 and targets.shape[1] > 2:
        func = multi_label_f1
    elif targets.ndim == 2 and targets.shape[1] == 2:
        func = binary_f1
    else:
        func = multi_class_f1
    value = func(targets, probs, threshold=0.5, average="macro")
    return name, value, higher_better


def precision(probs, labeled_data):
    higher_better = True
    name = "precision"
    targets = labeled_data.get_label()
    if targets.ndim == 1:
        if len(np.unique(targets)) == 2:
            func = binary_precision
        else:
            func = multi_class_precision
    elif targets.ndim == 2 and targets.shape[1] > 2:
        func = multi_label_precision
    elif targets.ndim == 2 and targets.shape[1] == 2:
        func = binary_precision
    else:
        func = multi_class_precision
    value = func(targets, probs, threshold=0.5, average="macro")
    return name, value, higher_better


def recall(probs, labeled_data):
    higher_better = True
    name = "recall"
    targets = labeled_data.get_label()
    if targets.ndim == 1:
        if len(np.unique(targets)) == 2:
            func = binary_recall
        else:
            func = multi_class_recall
    elif targets.ndim == 2 and targets.shape[1] > 2:
        func = multi_label_recall
    elif targets.ndim == 2 and targets.shape[1] == 2:
        func = binary_recall
    else:
        func = multi_class_recall
    value = func(targets, probs, threshold=0.5, average="macro")
    return name, value, higher_better


def roc_auc(probs, labeled_data):
    higher_better = True
    name = "roc_auc"
    targets = labeled_data.get_label()
    if targets.ndim == 1:
        if len(np.unique(targets)) == 2:
            func = binary_roc_auc
        else:
            func = multi_class_roc_auc
    elif targets.ndim == 2 and targets.shape[1] > 2:
        func = multi_label_roc_auc
    elif targets.ndim == 2 and targets.shape[1] == 2:
        func = binary_roc_auc
    else:
        func = multi_class_roc_auc
    value = func(labeled_data.get_label(), probs, threshold=0.5, average="macro")
    return name, value, higher_better


def pr_auc(probs, labeled_data):
    higher_better = True
    name = "pr_auc"
    targets = labeled_data.get_label()
    if targets.ndim == 1:
        if len(np.unique(targets)) == 2:
            func = binary_pr_auc
        else:
            func = multi_class_pr_auc
    elif targets.ndim == 2 and targets.shape[1] > 2:
        func = multi_label_pr_auc
    elif targets.ndim == 2 and targets.shape[1] == 2:
        func = binary_pr_auc
    else:
        func = multi_class_pr_auc
    value = func(targets, probs, threshold=0.5, average="macro")
    return name, value, higher_better


def save_model(model, model_path):
    print("#" * 25 + "save model" + "#" * 25)
    # dump model with pickle
    with open(model_path, "wb") as fout:
        pickle5.dump(model, fout)


def load_model(model_path):
    print("#" * 25 + "load model" + "#" * 25)
    with open(model_path, "rb") as fin:
        model = pickle5.load(fin)
        return model


def get_model_param(args):
    # LGBM Paramater tuning
    if args.task_type in ["binary_class", "binary-class"]:
        objective = "binary"
    elif args.task_type in ["multi_class", "multi-class"]:
        objective = "multiclass"
    elif args.task_type in ["multi_label", "multi-label"]:
        pass # to do
    elif args.task_type == "regression":
        objective = "regression" #  regression, regression_l1， mape

    '''
    param = {
        'num_leaves': 8,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'max_depth': 17,
        'objective': objective, # task type binary, regression, regression_l1， mape， multiclass
        'boosting_type': args.boosting_type, # gbdt, rf, dart, goss
        #'metric': ['auc', 'binary'],
        # 'metric': ['binary_error'],
        #'metric': 'custom',
        'metric': ['auc', 'binary'],
        "verbose": 1
    }
    '''
    config = json.load(open(args.config_path, "r"))
    config["objective"] = objective
    config["learning_rate"] = args.learning_rate
    if args.unbalance:
        config["is_unbalance"] = True
    assert config["boosting_type"] in ["gbdt", "rf", "dart", "goss"]
    return config


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

    X_train, y_train, label_list = load_dataset(args, "train")
    X_dev, y_dev, _ = load_dataset(args, "dev")
    X_test, y_test, _ = load_dataset(args, "test")
    if args.unbalance:
        train_dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False).construct()
        dev_dataset = lgb.Dataset(X_dev, label=y_dev, free_raw_data=False).construct()
        test_dataset = lgb.Dataset(X_test, label=y_test, free_raw_data=False).construct()
    else:
        if args.grid_search:
            train_dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False).construct()
            dev_dataset = lgb.Dataset(X_dev, label=y_dev, free_raw_data=False).construct()
            test_dataset = lgb.Dataset(X_test, label=y_test, free_raw_data=False).construct()
        else:
            train_weights = np.ones_like(y_train)
            train_weights[y_train == 1] = args.pos_weight
            dev_weights = np.ones_like(y_dev)
            dev_weights[y_dev == 1] = args.pos_weight
            test_weights = np.ones_like(y_test)
            test_weights[y_test == 1] = args.pos_weight
            train_dataset = lgb.Dataset(X_train, label=y_train, weight=train_weights, free_raw_data=False).construct()
            dev_dataset = lgb.Dataset(X_dev, label=y_dev, weight=dev_weights, free_raw_data=False).construct()
            test_dataset = lgb.Dataset(X_test, label=y_test, weight=test_weights, free_raw_data=False).construct()

    args.num_labels = len(np.unique(y_train))
    # output training/evaluation hyperparameters into logger
    logger.info("==== Training/Evaluation Parameters: =====")
    for attr, value in sorted(args.__dict__.items()):
        logger.info("\t{}={}".format(attr, value))
    logger.info("==== Parameters End =====\n")

    args_dict = {}
    for attr, value in sorted(args.__dict__.items()):
        if attr != "device":
            args_dict[attr] = value
    log_fp.write(json.dumps(args_dict, ensure_ascii=False) + "\n")
    log_fp.write("#" * 50 + "\n")

    if args.task_type in ["binary_class", "binary-class"]:
        n_train_pos = y_train.sum()
        n_dev_pos = y_dev.sum()
        n_test_pos = y_test.sum()
        logger.info("Train data, positive: {}, negative: {}, positive ratio: {:.2f}".format(n_train_pos, len(y_train) - n_train_pos, n_train_pos / len(y_train)))
        logger.info("Dev data, positive: {}, negative: {}, positive ratio: {:.2f}".format(n_dev_pos, len(y_dev) - n_train_pos, n_dev_pos / len(y_dev)))
        logger.info("Test data, positive: {}, negative: {}, positive ratio: {:.2f}".format(n_test_pos, len(y_test) - n_test_pos, n_test_pos / len(y_test)))
        log_fp.write("Train data, positive: {}, negative: {}, positive ratio: {:.2f}\n".format(n_train_pos, len(y_train) - n_train_pos, n_train_pos / len(y_train)))
        log_fp.write("Dev data, positive: {}, negative: {}, positive ratio: {:.2f}\n".format(n_dev_pos, len(y_dev) - n_train_pos, n_dev_pos / len(y_dev)))
        log_fp.write("Test data, positive: {}, negative: {}, positive ratio: {:.2f}\n".format(n_test_pos, len(y_test) - n_test_pos, n_test_pos / len(y_test)))
        log_fp.write("#" * 50 + "\n")

    log_fp.write("num labels: %d\n" % args.num_labels)
    log_fp.write("#" * 50 + "\n")
    log_fp.flush()

    max_metric_model_info = None
    if args.do_train:
        logger.info("++++++++++++Training+++++++++++++")
        if args.grid_search:
            global_step, tr_loss, max_metric_model_info = train_with_grid_search(args, train_dataset, dev_dataset, log_fp=log_fp)
        else:
            global_step, tr_loss, max_metric_model_info = trainer(args, train_dataset, dev_dataset, log_fp=log_fp)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # save
    if args.do_train:
        logger.info("++++++++++++Save Model+++++++++++++")
        # Create output directory if needed
        global_step = max_metric_model_info["global_step"]
        logger.info("max %s global step: %d" % (args.max_metric_type, global_step))
        log_fp.write("max %s global step: %d\n" % (args.max_metric_type, global_step))
        prefix = "checkpoint-{}".format(global_step)
        checkpoint = os.path.join(args.output_dir, prefix)
        logger.info("Saving model checkpoint to %s", checkpoint)
        torch.save(args, os.path.join(checkpoint, "training_args.bin"))
        save_labels(os.path.join(checkpoint, "label.txt"), label_list)

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
        lgbm_model = load_model(os.path.join(checkpoint, "lgbm_model.txt"))
        result = evaluate(args, lgbm_model, dev_dataset, prefix=prefix, log_fp=log_fp)
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
        lgbm_model = load_model(os.path.join(checkpoint, "lgbm_model.txt"))
        result = predict(args, lgbm_model, test_dataset, prefix=prefix, log_fp=log_fp)
        result = dict(("evaluation_" + k + "_{}".format(global_step), v) for k, v in result.items())
        logger.info(json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    log_fp.close()


def train_with_grid_search(args, train_dataset, dev_dataset, log_fp=None):
    parameters = {
        "num_leaves": [8, 12, 16, 20],
        "max_depth": [9, 15, 17, 20, 25, 30, 35, 39],
        "min_split_gain": [0, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        "subsample": [0.6, 0.7, 0.8, 0.8, 1],
        "n_estimators": [50, 100, 200, 250, 300],
        "learning_rate": [0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
        "feature_fraction": [0.6, 0.7, 0.8, 0.9, 0.95],
        "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 0.95],
        "bagging_freq": [2, 4, 5, 6, 8],
        "lambda_l1": [0, 0.1, 0.4, 0.5, 0.6],
        "lambda_l2": [0, 10, 15, 35, 40],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1],
        "reg_alpha": [0, 0.01, 0.02, 0.05, 0.09, 0.1, 1],
        "reg_lambda": [0, 0.1, 0.5, 1],
        "cat_smooth": [1, 10, 15, 20, 35]
    }
    if args.max_metric_type == "acc":
        feval = "accuracy"
    elif args.max_metric_type == "prec":
        feval = "precision"
    elif args.max_metric_type == "recall":
        feval = "recall"
    elif args.max_metric_type == "f1":
        feval = "f1_macro"
    elif args.max_metric_type == "pr_auc":
        feval = "average_precision"
    elif args.max_metric_type == "roc_auc":
        feval = "roc_auc"
    else:
        feval = "f1_macro"
    param = get_model_param(args)
    if args.output_mode in ["binary_class", "binary-class"]:
        class_weight = {0: 1, 1: args.pos_weight}
    else:
        class_weight = None
    gbm = LGBMClassifier(
        boosting_type=param["boosting_type"],
        num_leaves=param["num_leaves"],
        max_depth=param["max_depth"],
        learning_rate=param["learning_rate"],
        n_estimators=100,
        subsample_for_bin=200000,
        objective=param["objective"],
        num_class=1 if args.num_labels == 2 else args.num_labels,
        class_weight=class_weight,
        min_split_gain=0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=args.seed,
        n_jobs=-1,
        silent="warn",
        importance_type="split"
    )

    # use gridsearch without fit function
    train_val_features = np.concatenate((train_dataset.get_data(), dev_dataset.get_data()), axis=0)
    train_val_labels = np.concatenate((train_dataset.get_label(), dev_dataset.get_label()), axis=0)

    dev_fold = np.zeros(train_val_features.shape[0])
    dev_fold[:train_dataset.get_data().shape[0]] = -1
    ps = PredefinedSplit(test_fold=dev_fold)

    # gsearch = GridSearchCV(gbm, param_grid=parameters, scoring=feval, cv=ps)
    gsearch = RandomizedSearchCV(gbm, param_distributions=parameters, scoring=feval, cv=ps)
    gsearch.fit(train_val_features,
                train_val_labels,
                eval_set=[(dev_dataset.get_data(), dev_dataset.get_label())],
                eval_metric=['auc', 'binary_logloss'],
                early_stopping_rounds=args.early_stopping_rounds if args.early_stopping_rounds > 0 else None,
                verbose=1)

    logger.info("Best score[%s]: %0.6f" % (args.max_metric_type, gsearch.best_score_))
    logger.info("Best parameters set:")
    log_fp.write("Best score[%s]: %0.6f\n" % (args.max_metric_type, gsearch.best_score_))
    log_fp.write("Best parameters set:\n")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))
        log_fp.write("%s: %r\n" % (param_name, best_parameters[param_name]))
    log_fp.write("#" * 50 + "\n")
    global_step = best_parameters["n_estimators"]
    prefix = "checkpoint-{}".format(global_step)
    checkpoint = os.path.join(args.output_dir, prefix)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    # save gridsearch
    joblib.dump(gsearch, os.path.join(args.output_dir, "lgbm_gridsearch.pkl"))

    log_fp.write(str(gsearch.best_estimator_) + "\n" + "#" * 50 + "\n")

    tr_loss = 0
    max_metric_model_info = {"global_step": global_step}
    save_model(gsearch.best_estimator_, os.path.join(checkpoint, "lgbm_model.txt"))
    json.dump(best_parameters, open(os.path.join(checkpoint, "config.json"), "w"))
    return global_step, tr_loss, max_metric_model_info


def trainer(args, train_dataset, dev_dataset, log_fp=None):
    if log_fp is None:
        log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    feature_size = train_dataset.num_feature()
    feature_names = ["f_%d" % idx for idx in range(feature_size)]
    train_num = train_dataset.num_data()
    if args.max_metric_type == "acc":
        feval = acc
    elif args.max_metric_type == "prec":
        feval = precision
    elif args.max_metric_type == "recall":
        feval = recall
    elif args.max_metric_type == "f1":
        feval = f1
    elif args.max_metric_type == "pr_auc":
        feval = pr_auc
    elif args.max_metric_type == "roc_auc":
        feval = roc_auc
    else:
        feval = f1
    param = get_model_param(args)

    log_fp.write(json.dumps(param, ensure_ascii=False) + "\n")
    log_fp.write("#" * 50 + "\n")
    log_fp.flush()

    evals_result = {}
    lgbm_model = lgb.train(
        params=param,
        train_set=train_dataset,
        num_boost_round=args.num_train_epochs,
        valid_sets=[train_dataset, dev_dataset],
        valid_names=["train", "dev"],
        callbacks=[lgb.record_evaluation(evals_result)],
        feval=feval,
        feature_name=feature_names,
        early_stopping_rounds=args.early_stopping_rounds if args.early_stopping_rounds > 0 else None,
        verbose_eval=1,
        keep_training_booster=True
    )

    feature_importance_info = pd.DataFrame()
    feature_importance_info["feature"] = feature_names
    feature_importance_info["importance"] = lgbm_model.feature_importance()

    log_fp.write("best iteration: %d\n" % lgbm_model.best_iteration)
    log_fp.write("num trees: %d\n" % lgbm_model.num_trees())
    global_step = lgbm_model.best_iteration
    prefix = "checkpoint-{}".format(global_step)
    checkpoint = os.path.join(args.output_dir, prefix)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    log_fp.write(str(evals_result) + "\n" + "#" * 50 + "\n")
    tr_loss = evals_result["train"]["binary_logloss"][global_step - 1]
    max_metric_model_info = {"global_step": global_step}
    save_model(lgbm_model, os.path.join(checkpoint, "lgbm_model.txt"))
    param["best_iteration"] = lgbm_model.best_iteration
    json.dump(param, open(os.path.join(checkpoint, "config.json"), "w"))
    return global_step, tr_loss, max_metric_model_info


def evaluate(args, model, dataset, prefix, log_fp=None):
    X, y = dataset.get_data(), dataset.get_label()
    dev_num = X.shape[0]
    if args.grid_search:
        hat_y = model.predict_proba(X)
    else:
        hat_y = model.predict(X)
    if args.output_mode in ["multi-label", "multi_label"]:
        result = metrics_multi_label(y, hat_y, threshold=0.5)
    elif args.output_mode in ["multi-class", "multi_class"]:
        result = metrics_multi_class(y, hat_y)
    elif args.output_mode == "regression":
        pass # to do
    elif args.output_mode in ["binary-class", "binary_class"]:
        result = metrics_binary(y, hat_y, threshold=0.5,
                                savepath=os.path.join(args.output_dir, "dev_confusion_matrix.png"))

    with open(os.path.join(args.output_dir, "dev_metrics.txt"), "w") as writer:
        logger.info("***** Eval Dev results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    logger.info("Dev metrics: ")
    logger.info(json.dumps(result, ensure_ascii=False))
    logger.info("")
    return result


def predict(args, model, dataset, prefix, log_fp=None):
    X, y = dataset.get_data(), dataset.get_label()
    test_num = X.shape[0]
    if args.grid_search:
        hat_y = model.predict_proba(X)
    else:
        hat_y = model.predict(X)
    if args.output_mode in ["multi_class", "multi-class"]:
        result = metrics_multi_class(y, hat_y)
    elif args.output_mode in ["multi_label", "multi-label"]:
        result = metrics_multi_label(y, hat_y, threshold=0.5)
    elif args.output_mode == "regression":
        pass # to do
    elif args.output_mode in ["binary_class", "binary-class"]:
        result = metrics_binary(y, hat_y, threshold=0.5,
                                savepath=os.path.join(args.output_dir, "test_confusion_matrix.png"))

    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as wfp:
        for idx in range(len(y)):
            wfp.write("%d,%s,%s\n" %(idx, str(hat_y[idx]), str(y[idx])))

    with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as wfp:
        logger.info("***** Eval Test results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            wfp.write("%s = %s\n" % (key, str(result[key])))

    logger.info("Test metrics: ")
    logger.info(json.dumps(result, ensure_ascii=False))
    logger.info("")
    return result


if __name__ == "__main__":
    run()
