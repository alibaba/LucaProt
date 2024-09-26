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
@datetime: 2023/03/10 20:18
@project: DeepProtFunc
@file: predict
@desc: predict batch data from file
'''

import argparse, time
import numpy as np
import os, sys, csv
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
sys.path.append("../../src/common")
try:
    from deep_baselines.cheer import CatWCNN, WDCNN, WCNN, seq_encode as cheer_seq_encode
    from deep_baselines.virhunter import VirHunter, seq_encode as virhunter_seq_encode, one_hot_encode as virhunter_seq_hot_encode
    from deep_baselines.virtifier import Virtifier, seq_encode as virtifier_seq_encode
    from deep_baselines.virseeker import VirSeeker, seq_encode as virseeker_seq_encode
    from common.metrics import *
    from common.multi_label_metrics import *
    from utils import set_seed, plot_bins, csv_reader
except ImportError:
    from src.deep_baselines.cheer import CatWCNN, WDCNN, WCNN, seq_encode as cheer_seq_encode
    from src.deep_baselines.virhunter import VirHunter, seq_encode as virhunter_seq_encode, one_hot_encode as virhunter_seq_hot_encode
    from src.deep_baselines.virtifier import Virtifier, seq_encode as virtifier_seq_encode
    from src.deep_baselines.virseeker import VirSeeker, seq_encode as virseeker_seq_encode
    from src.common.metrics import *
    from src.common.multi_label_metrics import *
    from src.utils import set_seed, plot_bins, csv_reader

import logging
logger = logging.getLogger(__name__)


def llprint(message):
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def load_label_code_2_name(args, filename):
    '''
    load the mapping between the label name and label code
    :param args:
    :param filename:
    :return:
    '''
    label_code_2_name = {}
    filename = "../../dataset/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type, filename)
    if filename and os.path.exists(filename):
        with open(filename, "r") as rfp:
            for line in rfp:
                strs = line.strip().split("###")
                label_code_2_name[strs[0]] = strs[1]
    return label_code_2_name


def load_args(model_dir):
    '''
    load model running args
    :param model_dir:
    :return: config
    '''
    print("model_dir: ", model_dir)
    args_filepath = os.path.join(model_dir, "training_args.bin")
    if not os.path.exists(args_filepath):
        raise Exception("%s not exists" % args_filepath)
    args = torch.load(args_filepath)
    new_args = {}
    for attr, value in sorted(args.__dict__.items()):
        new_args[attr] = value
    return new_args


def load_model(args, model_dir):
    '''
    load the model
    :param args:
    :param model_dir:
    :return:
    '''
    # load tokenizer and model
    device = torch.device(args.device)
    '''
    if args.model_type == "CHEER-CatWCNN":
        model_class = CatWCNN
    elif args.model_type == "CHEER-WDCNN":
        model_class = WDCNN
    elif args.model_type == "CHEER-WCNN":
        model_class = WCNN
    elif args.model_type == "VirHunter":
        model_class = VirHunter
    elif args.model_type == "Virtifier":
        model_class = Virtifier
    elif args.model_type == "VirSeeker":
        model_class = VirSeeker
    else:
        raise Exception("not support model type: %s" % args.model_type)
    model = model_class.from_pretrained(model_dir, args=args)
    '''
    model = torch.load(os.path.join(model_dir, "%s.pkl" %args.model_type))

    model.to(device)
    model.eval()
    # load labels
    if not os.path.exists(args.label_filepath):
        args.label_filepath = os.path.join("../../dataset/%s/%s/%s/%s" %(args.dataset_name, args.dataset_type, args.task_type, "label.txt"))
    label_filepath = args.label_filepath
    label_id_2_name = {}
    label_name_2_id = {}
    with open(label_filepath, "r") as fp:
        for line in fp:
            if line.strip() == "label":
                continue
            label_name = line.strip()
            label_id_2_name[len(label_id_2_name)] = label_name
            label_name_2_id[label_name] = len(label_name_2_id)
    print("-----------label_id_2_name:------------")
    if len(label_id_2_name) < 20:
        print(label_id_2_name)
    print("label size: ", len(label_id_2_name))

    return model, label_id_2_name, label_name_2_id


def load_vocab(vocab_filepath):
    token_list = []
    with open(vocab_filepath, "r") as rfp:
        for line in rfp:
            token_list.append(line.strip())
    int_to_token = {idx: token for idx, token in enumerate(token_list)}
    token_to_int = {token: idx for idx, token in int_to_token.items()}
    return int_to_token, token_to_int


def transform_sample_2_feature(args, rows, token_2_int):
    '''
    batch sample transform to batch input
    :param args:
    :param rows: [[prot_id, protein_seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename], ...]
    :return:
    '''
    batch_info = []
    x = []
    lens = []
    for row in rows:
        # agreed 7 columns
        if len(row) < 7:
            prot_id, protein_seq = row[0:2]
            batch_info.append([prot_id, protein_seq])
        else:
            prot_id, protein_seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename = row[0:7]
            batch_info.append([prot_id, protein_seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename])
        if len(row) > 7:
            batch_info[-1].extend(row[7:])
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
        encode_func_args["seq"] = protein_seq.upper()
        seq_ids, actural_len = encode_func(**encode_func_args)

        x.append(seq_ids)
        lens.append(actural_len)

    return batch_info, {"x": torch.tensor(x, dtype=torch.long).to(args.device), "lengths": torch.tensor(lens, dtype=torch.long).to(args.device)}


def predict_probs(
        args,
        model,
        token_2_int,
        rows
):
    '''
    prediction
    :param args:
    :param model:
    :param rows:
    :return:
    '''
    '''
    label_list = processor.get_labels(label_filepath=args.label_filepath)
    label_map = {label: i for i, label in enumerate(label_list)}
    '''
    batch_info, batch_input = transform_sample_2_feature(args, rows, token_2_int)
    if torch.cuda.is_available():
        probs = model(**batch_input)[1].detach().cpu().numpy()
    else:
        probs = model(**batch_input)[1].detach().numpy()
    return batch_info, probs


def predict_binary_class(
        args,
        label_id_2_name,
        token_2_int,
        model,
        rows
):
    '''
    predict positive or negative label for a batch
    :param args:
    :param label_id_2_name:
    :param model:
    :param rows: n samples
    :return:
    '''
    batch_info, probs = predict_probs(args, model, token_2_int, rows)
    # print("probs dim: ", probs.ndim)
    preds = (probs >= args.threshold).astype(int).flatten()
    res = []
    for idx, info in enumerate(batch_info):
        res.append([info[0], info[1], float(probs[idx][0]), label_id_2_name[preds[idx]], *info[2:]])
    return res


def predict_multi_class(
        args,
        label_id_2_name,
        token_2_int,
        model,
        rows
):
    '''
    predict a class for a batch
    :param args:
    :param label_id_2_name:
    :param model:
    :param rows: n samples
    :return:
    '''
    batch_info, probs = predict_probs(args, model, token_2_int, rows)
    # print("probs dim: ", probs.ndim)
    preds = np.argmax(probs, axis=-1)
    res = []
    for idx, info in enumerate(batch_info):
        res.append([info[0], info[1], float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]], *info[2:]])
    return res


def predict_multi_label(
        args,
        label_id_2_name,
        token_2_int,
        model,
        rows
):
    '''
    predict multi-labels for a batch
    :param args:
    :param label_id_2_name:
    :param model:
    :param rows: n samples
    :return:
    '''
    batch_info, probs = predict_probs(args, model, token_2_int, rows)
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    res = []
    for idx, info in enumerate(batch_info):
        res.append([info[0], info[1], [float(probs[idx][label_index]) for label_index in preds[idx]],
                    [label_id_2_name[label_index] for label_index in preds[idx]], *info[2:]])
    return res


parser = argparse.ArgumentParser(description="Prediction")
parser.add_argument("--data_path", default=None, type=str, required=True,
                    help="the data filepath(if it is csv format, Column 0 must be id, Colunm 1 must be seq.")
parser.add_argument("--dataset_name", default="rdrp_40_extend", type=str, required=True,
                    help="the dataset name for model building.")
parser.add_argument("--dataset_type", default="protein", type=str, required=True,
                    help="the dataset type for model building.")
parser.add_argument("--task_type", default=None, type=str, required=True,
                    choices=["multi_label", "multi_class", "binary_class"],
                    help="the task type for model building.")
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="the model type.")
parser.add_argument("--time_str", default=None, type=str, required=True,
                    help="the running time string(yyyymmddHimiss) of model building.")
parser.add_argument("--step", default=None, type=str, required=True,
                    help="the training global step of model finalization.")
parser.add_argument("--evaluate", action="store_true",
                    help="whether to evaluate the predicted results.")
parser.add_argument("--ground_truth_col_index",  default=None, type=int,
                    help="the ground truth col index of the ${data_path}, default: None.")
parser.add_argument("--threshold",  default=0.5, type=float,
                    help="sigmoid threshold for binary-class or multi-label classification, None for multi-class classification, defualt: 0.5.")
parser.add_argument("--batch_size",  default=16, type=int,
                    help="batch size per GPU/CPU for evaluation, default: 16.")
parser.add_argument("--print_per_batch",  default=1000, type=int,
                    help="how many batches are completed every time for printing progress information, default: 1000.")
args = parser.parse_args()


if __name__ == "__main__":
    model_dir = "../../models/%s/%s/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type,
                                                    args.model_type, args.time_str,
                                                    args.step if args.step == "best" else "checkpoint-{}".format(args.step))
    config_dir = "../../logs/%s/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type,
                                                args.model_type,  args.time_str)
    predict_dir = "../../predicts/%s/%s/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type,
                                                        args.model_type, args.time_str,
                                                        "checkpoint-{}".format(args.step))

    # Step1: loading the model configuration
    config = load_args(model_dir)
    print("-" * 25 + "config:" + "-" * 25)
    print(config)
    if config:
        args.dataset_name = config["dataset_name"]
        args.dataset_type = config["dataset_type"]
        args.task_type = config["task_type"]
        args.model_type = config["model_type"]
        args.label_filepath = config["label_filepath"]
        if not os.path.exists(args.label_filepath):
            args.label_filepath = os.path.join(config_dir, "label.txt")
        args.output_dir = config["output_dir"]
        args.config_path = config["config_path"]

        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        args.seq_vocab_path = config["seq_vocab_path"]
        args.seq_max_length = config["seq_max_length"]
        args.trunc_type = config["trunc_type"]
        args.one_hot_encode = config["one_hot_encode"]
        if "channel_in" in config:
            args.channel_in = config["channel_in"]

        if args.task_type in ["multi-label", "multi_label"]:
            # to do
            args.sigmoid = True
        elif args.task_type in ["binary-class", "binary_class"]:
            args.sigmoid = True
    print("-" * 25 + "args:" + "-" * 25)
    print(args.__dict__.items())
    print("-" * 25 + "model_dir list:" + "-" * 25)
    print(os.listdir(model_dir))

    # Step2: loading the tokenizer and model
    model, label_id_2_name, label_name_2_id = load_model(args, model_dir)
    int_2_token, token_2_int = load_vocab(args.seq_vocab_path)
    # config.vocab_size = len(token_2_int)

    predict_func = None
    evaluate_func = None
    if args.task_type in ["multi-label", "multi_label"]:
        predict_func = predict_multi_label
        evaluate_func = metrics_multi_label_for_pred
    elif args.task_type in ["binary-class", "binary_class"]:
        predict_func = predict_binary_class
        evaluate_func = metrics_binary_for_pred
    elif args.task_type in ["multi-class", "multi_class"]:
        predict_func = predict_multi_class
        evaluate_func = metrics_multi_class_for_pred
    # append other columns to the result csv file
    if args.data_path.endswith(".csv"):
        # expand the csv header
        col_names = ["protein_id", "seq", "predict_prob", "predict_label", "seq_len", "pdb_filename", "ptm", "mean_plddt", "emb_filename"]
        with open(args.data_path, "r") as rfp:
            for row in csv_reader(rfp, header=True, header_filter=False):
                if len(row) > 7:
                    col_names.extend(row[7:])
                break
    elif args.data_path.endswith(".fasta") or args.data_path.endswith(".fas") or args.data_path.endswith(".fa"):
        col_names = ["protein_id", "seq", "predict_prob", "predict_label"]
    else:
        raise Exception("not csv file or fasta file.")

    # statistics
    ground_truth_stats = {}
    predict_stats = {}
    seq_len_stats = {}

    # the result savepath
    parent_dirname = ".".join(os.path.basename(args.data_path).split(".")[0:-1])
    predict_dir = os.path.join(predict_dir, parent_dirname)
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)

    # what has already been done does not require prediction
    pred_result_path = os.path.join(predict_dir, "pred_result.csv")
    write_mode = "w"
    had_done_cnt = 0
    total_cnt = 0
    if os.path.exists(pred_result_path):
        with open(pred_result_path, "r") as rfp:
            for row in csv_reader(rfp, header=True, header_filter=True):
                had_done_cnt += 1
                predict_label_name = row[3]
                if args.task_type in ["multi-label", "multi_label"]:
                    predict_label_name = eval(predict_label_name)
                if isinstance(predict_label_name, list):
                    for v in predict_label_name:
                        if v not in predict_stats:
                            predict_stats[v] = 1
                        else:
                            predict_stats[v] += 1
                else:
                    if predict_label_name not in predict_stats:
                        predict_stats[predict_label_name] = 1
                    else:
                        predict_stats[predict_label_name] += 1
        if had_done_cnt > 0:
            write_mode = "a+"
    # total records number
    rfp = open(args.data_path, "r")
    if args.data_path.endswith(".csv"):
        reader = csv_reader(rfp, header=True, header_filter=True)
        for row in reader:
            total_cnt += 1
    elif args.data_path.endswith(".fasta") or args.data_path.endswith(".fas") or args.data_path.endswith(".fa"):
        for row in fasta_reader(args.data_path):
            total_cnt += 1
    else:
        raise Exception("not csv file or fasta file.")
    # total batch number
    total_batch_num = (total_cnt + args.batch_size - 1 - had_done_cnt)//args.batch_size
    print("total num: %d, had done num: %d, batch size: %d, batch_num: %d" %(total_cnt, had_done_cnt, args.batch_size, total_batch_num))
    # Step 3: prediction
    with open(pred_result_path,  write_mode) as wfp:
        writer = csv.writer(wfp)
        # not keep running
        if write_mode == "w":
            writer.writerow(col_names)
        # The number of batches that have been predicted
        done_batch_num = 0
        rfp = open(args.data_path, "r")
        if args.data_path.endswith(".csv"):
            reader = csv_reader(rfp, header=True, header_filter=True)
        elif args.data_path.endswith(".fasta") or args.data_path.endswith(".fas") or args.data_path.endswith(".fa"):
            reader = fasta_reader(args.data_path)
        else:
            raise Exception("not csv file.")
        row_batch = []
        cur_cnt = 0
        use_time = 0
        for row in reader:
            # prot_id, protein_seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename
            seq_len = len(row[1])
            if seq_len not in seq_len_stats:
                seq_len_stats[seq_len] = 1
            else:
                seq_len_stats[seq_len] += 1
            cur_cnt += 1
            # If the had_done_cnt item has been predicted, the prediction will not be repeated
            if cur_cnt <= had_done_cnt:
                continue
            # one batch
            row_batch.append(row)
            # there is ground truth, then calc the statistics
            if args.ground_truth_col_index:
                ground_truth_label_name = row[args.ground_truth_col_index]
                if ground_truth_label_name not in ground_truth_stats:
                    ground_truth_stats[ground_truth_label_name] = 1
                else:
                    ground_truth_stats[ground_truth_label_name] += 1

            # predict one batch
            if len(row_batch) % args.batch_size == 0:
                begin_time = time.time()
                res = predict_func(args, label_id_2_name, token_2_int, model, row_batch)
                use_time += time.time() - begin_time
                # Traverse one batch results
                for item in res:
                    predict_label_name = item[3]
                    if isinstance(predict_label_name, list): # multi-label
                        for v in predict_label_name:
                            if v not in predict_stats:
                                predict_stats[v] = 1
                            else:
                                predict_stats[v] += 1
                    else:
                        if predict_label_name not in predict_stats:
                            predict_stats[predict_label_name] = 1
                        else:
                            predict_stats[predict_label_name] += 1

                    writer.writerow(item)
                row_batch = []
                done_batch_num += 1
                if done_batch_num % args.print_per_batch == 0:
                    llprint("batch: %10d, done rate: %0.2f%%" % (done_batch_num, done_batch_num * 100/total_batch_num))
                    print("done total: %d, p: %d, n: %d, per batch use time: %f" % (done_batch_num * args.batch_size + had_done_cnt,
                                                                                    predict_stats["1"] if "1" in predict_stats else (predict_stats[1] if 1 in predict_stats else 0),
                                                                                    predict_stats["0"] if "0" in predict_stats else (predict_stats[0] if 0in predict_stats else 0),
                                                                                    use_time/done_batch_num))
        if len(row_batch) > 0:
            begin_time = time.time()
            res = predict_func(args, label_id_2_name, token_2_int, model, row_batch)
            use_time += time.time() - begin_time
            row_batch = []
            for item in res:
                predict_label_name = item[3]
                if isinstance(predict_label_name, list):
                    for v in predict_label_name:
                        if v not in predict_stats:
                            predict_stats[v] = 1
                        else:
                            predict_stats[v] += 1
                else:
                    if predict_label_name not in predict_stats:
                        predict_stats[predict_label_name] = 1
                    else:
                        predict_stats[predict_label_name] += 1
                writer.writerow(item)
            done_batch_num += 1
            llprint("batch: %10d, done rate: %0.2f%%" %(done_batch_num, done_batch_num*100/total_batch_num))
            print("done total: %d, p: %d, n: %d, per batch use time: %f" % (total_cnt,
                                                                            predict_stats["1"] if "1" in predict_stats else (predict_stats[1] if 1 in predict_stats else 0),
                                                                            predict_stats["0"] if "0" in predict_stats else (predict_stats[0] if 0in predict_stats else 0),
                                                                            use_time/done_batch_num))
        print("prediction done. total batch: %d, use time: %f." % (done_batch_num, use_time))

    # plot the Sequence Length Distribution
    seq_length_distribution_pic_savepath = os.path.join(predict_dir, "seq_length_distribution.png")
    if not os.path.exists(os.path.dirname(seq_length_distribution_pic_savepath)):
        os.makedirs(os.path.dirname(seq_length_distribution_pic_savepath))
    seq_len_list = []
    for item in seq_len_stats.items():
        seq_len_list.extend([item[0]] * item[1])
    plot_bins(seq_len_list, xlabel="sequence length", ylabel="distribution", bins=40, filepath=seq_length_distribution_pic_savepath)

    # calc metrics
    evaluate_metrics = None
    predict_metrics_savepath = os.path.join(predict_dir, "pred_metrics.txt")
    label_size = len(label_id_2_name)

    # if there is ground truth, all metrics can be calculated
    if args.evaluate and args.ground_truth_col_index:
        ground_truth_stats = {}
        confusion_matrix_savepath = os.path.join(predict_dir, "pred_confusion_matrix.png")
        ground_truth_list = []
        predict_pred_list = []
        with open(os.path.join(predict_dir, "pred_result.csv"), "r") as rfp:
            reader = csv_reader(rfp, header=True, header_filter=True)
            for row in reader:
                predict_prob = row[2]
                predict_label = row[3]
                # because the file has added two columns (the third column and the fourth column), the column number is increased by 2
                ground_truth = row[args.ground_truth_col_index + 2]
                if ground_truth not in ground_truth_stats:
                    ground_truth_stats[ground_truth] = 1
                else:
                    ground_truth_stats[ground_truth] += 1
                # samples without ground_truth do not participate in the evaluation
                if ground_truth is None or len(ground_truth) == 0 or ground_truth in ["nono", "None"]:
                    continue
                if args.task_type in ["multi-label", "multi_label"]:
                    predict_label_id = [label_name_2_id[name] for name in eval(predict_label)]
                    ground_truth_id = [label_name_2_id[name] for name in eval(ground_truth)]
                    predict_label_indicator = label_id_2_array(predict_label_id, label_size)
                    ground_truth_indicator = label_id_2_array(ground_truth_id, label_size)
                    predict_pred_list.append(predict_label_indicator)
                    ground_truth_list.append(ground_truth_indicator)
                else:
                    predict_label_id = label_name_2_id[predict_label]
                    ground_truth_id = label_name_2_id[ground_truth]
                    predict_pred_list.append(predict_label_id)
                    ground_truth_list.append(ground_truth_id)

        evaluate_metrics = evaluate_func(np.array(ground_truth_list), np.array(predict_pred_list), savepath=confusion_matrix_savepath)
        print("predict metrics: ")
        print(evaluate_metrics)
    with open(predict_metrics_savepath, "w") as wfp:
        if evaluate_metrics:
            for key in sorted(evaluate_metrics.keys()):
                wfp.write("%s=%s\n" % (key, str(evaluate_metrics[key])))
            wfp.write("#" * 50 + "\n")
        elif label_size == 2:
            # binary classification, only one ground_truth(all are positive samples or negative samples)
            if len(ground_truth_stats) == 1:
                # calc the TP and TN
                if "1" in ground_truth_stats:
                    tp = predict_stats["1"]
                    fn = predict_stats["0"]
                    wfp.write("%s=%s\n" % ("tp", tp))
                    wfp.write("%s=%s\n" % ("fn", fn))
                elif "0" in ground_truth_stats:
                    # calc the TN and FP
                    tn = predict_stats["0"]
                    fp = predict_stats["1"]
                    wfp.write("%s=%s\n" % ("tn", tn))
                    wfp.write("%s=%s\n" % ("fp", fp))
        elif label_size > 2:
            # multi-class/mutil-label classification
            keys = ground_truth_stats.keys()
            if len(keys) == 1 and "" not in keys:
                # calc the TP and FN
                only_key = list(keys)[0]
                tp = predict_stats[only_key]
                fn = ground_truth_stats[only_key] - tp
                wfp.write("%s %s=%s\n" % (only_key, "tp", tp))
                wfp.write("%s %s=%s\n" % (only_key, "fn", fn))

        wfp.write("ground truth statistics:\n")
        for item in sorted(ground_truth_stats.items(), key=lambda x:x[0]):
            wfp.write("%s=%s\n" %(item[0], item[1]))
        wfp.write("#" * 50 + "\n")
        wfp.write("prediction statistics:\n")
        for item in sorted(predict_stats.items(), key=lambda x:x[0]):
            wfp.write("%s=%s\n" %(item[0], item[1]))
    print("-" * 25 + "predict stats:" + "-" * 25)
    print("ground truth: ")
    print(ground_truth_stats)
    print("prediction: ")
    print(predict_stats)



