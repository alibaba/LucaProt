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
@datetime: 2022/11/28 19:31
@project: DeepProtFunc
@file: utils
@desc: utils
'''
import os
import csv
import subprocess
import torch
import numpy as np
import sys, random
import io, textwrap, itertools
from Bio import SeqIO
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.unicode_minus'] = False
sys.path.append("..")
sys.path.append("../src")
try:
    from common.multi_label_metrics import prob_2_pred, relevant_indexes
except ImportError:
    from src.common.multi_label_metrics import prob_2_pred, relevant_indexes


def get_parameter_number(model):
    '''
    colc the parameter number of the model
    :param model: 
    :return: 
    '''
    param_size = 0
    param_sum = 0
    trainable_size = 0
    trainable_num = 0
    for param in model.parameters():
        cur_size = param.nelement() * param.element_size()
        cur_num = param.nelement()
        param_size += cur_size
        param_sum += cur_num
        if param.requires_grad:
            trainable_size += cur_size
            trainable_num += cur_num
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    '''
    total_num = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_num += sum(p.numel() for p in model.buffers())
    total_size += sum(p.numel() * p.element_size() for p in model.buffers())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    '''
    return {
        'total_num': "%fM" % round((buffer_sum + param_sum)/(1024 * 1024), 2),
        'total_size': "%fMB" % round((buffer_size + param_size)/(1024 * 1024), 2),
        'param_sum': "%fM" % round(param_sum/(1024 * 1024), 2),
        'param_size': "%fMB" % round(param_size/(1024 * 1024), 2),
        'buffer_sum': "%fM" % round(buffer_sum/(1024 * 1024), 2),
        'buffer_size': "%fMB" % round(buffer_size/(1024 * 1024), 2),
        'trainable_num': "%fM" % round(trainable_num/(1024 * 1024), 2),
        'trainable_size': "%fMB" % round(trainable_size/(1024 * 1024), 2)
    }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def file_reader(filename, header=True, header_filter=True):
    if filename.endswith(".csv"):
        return csv_reader(filename, header=header, header_filter=header_filter)
    elif filename.endswith(".tsv"):
        return tsv_reader(filename, header=header, header_filter=header_filter)
    else:
        return txt_reader(filename, header=header, header_filter=header_filter)


def txt_reader(handle, header=True, header_filter=True):
    '''
    text reader
    :param handle:
    :param header:
    :param header_filter: whether filter the header
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        cnt = 0
        for line in handle:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield line.strip()
    finally:
        if not handle.closed:
            handle.close()


def tsv_reader(handle, header=True, header_filter=True):
    '''
    tsv reader
    :param handle:
    :param header:
    :param header_filter: whether filter the header
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        reader = csv.reader(handle, delimiter="\t")
        cnt = 0
        for row in reader:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield row
    finally:
        if not handle.closed:
            handle.close()


def csv_reader(handle, header=True, header_filter=True):
    '''
    csv reader
    :param handle:
    :param header:
    :param header_filter: whether filter the header
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        reader = csv.reader(handle)
        cnt = 0
        for row in reader:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield row
    finally:
        if not handle.closed:
            handle.close()


def fasta_reader(handle, width=None):
    """
    Reads a FASTA file, yielding header, sequence pairs for each sequence recovered
    args:
        :handle (str, pathliob.Path, or file pointer) - fasta to read from
        :width (int or None) - formats the sequence to have max `width` character per line.
                               If <= 0, processed as None. If None, there is no max width.
    yields:
        :(header, sequence) tuples
    returns:
        :None
    """
    FASTA_STOP_CODON = "*"

    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    width = width if isinstance(width, int) and width > 0 else None
    try:
        header = None
        for is_header, group in itertools.groupby(handle, lambda line: line.startswith(">")):
            if is_header:
                header = group.__next__().strip()
            else:
                seq = ''.join(line.strip() for line in group).strip().rstrip(FASTA_STOP_CODON)
                if width is not None:
                    seq = textwrap.fill(seq, width)
                yield header, seq
    finally:
        if not handle.closed:
            handle.close()


def write_fasta(filepath, sequences):
    '''
    write fasta file
    :param filepath: savepath
    :param sequences: fasta sequence(each item: [id, seq])
    :return:
    '''
    with open(filepath, "w") as output_handle:
        for sequence in sequences:
            SeqIO.write(sequence, output_handle, "fasta")


def label_id_2_label_name(output_mode, label_list, prob, threshold=0.5):
    '''
    convect label id to label name
    :param output_mode:
    :param label_list:
    :param prob:
    :param threshold:
    :return:
    '''
    if output_mode in ["multi-label", "multi_label"]:
        res = []
        pred = prob_2_pred(prob, threshold)
        pred_index = relevant_indexes(pred)
        for row in range(prob.shape[0]):
            label_names = [label_list[idx] for idx in pred_index[row]]
            res.append(label_names)
        return res
    elif output_mode in ["multi-class", "multi_class"]:
        pred = np.argmax(prob, axis=1)
        label_names = [label_list[idx] for idx in pred]
        return label_names
    elif output_mode in ["binary-class", "binary_class"]:
        if prob.ndim == 2:
            prob = prob.flatten(order="C")
        pred = prob_2_pred(prob, threshold)
        label_names = [label_list[idx] for idx in pred]
        return label_names
    else:
        raise KeyError(output_mode)


def plot_bins(data, xlabel, ylabel, bins, filepath):
    '''
    plot bins
    :param data:
    :param xlabel:
    :param ylabel:
    :param bins: bins number
    :param filepath: png save filepath
    :return:
    '''
    plt.figure(figsize=(40, 20), dpi=100)
    plt.hist(data, bins=bins)
    # plt.xticks(range(min(data), max(data)))
    # plt.grid(linestyle='--', alpha=0.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.clf()
    plt.close()


def plot_confusion_matrix_for_binary_class(targets, preds, cm=None, savepath=None):
    '''
    :param targets: ground truth
    :param preds: prediction probs
    :param cm: confusion matrix
    :param savepath: confusion matrix picture savepth
    '''

    plt.figure(figsize=(40, 20), dpi=100)
    if cm is None:
        cm = confusion_matrix(targets, preds, labels=[0, 1])

    plt.matshow(cm, cmap=plt.cm.Oranges)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), verticalalignment='center', horizontalalignment='center')
    plt.ylabel('True')
    plt.xlabel('Prediction')
    if savepath:
        plt.savefig(savepath, dpi=100)
    else:
        plt.show()
    plt.close("all")


def save_labels(filepath, label_list):
    '''
    save labels
    :param filepath:
    :param label_list:
    :return:
    '''
    with open(filepath, "w") as wfp:
        wfp.write("label" + "\n")
        for label in label_list:
            wfp.write(label + "\n")


def load_labels(filepath, header=True):
    '''
    load labels
    :param filepath:
    :param header: where the file has header or not
    :return:
    '''
    label_list = []
    with open(filepath, "r") as rfp:
        for label in rfp:
            label_list.append(label.strip())
    if header and len(label_list) > 0:
        return label_list[1:]
    return label_list


def numpy_sparse_2_tenor_sparse(arr):
    import numpy as np
    import torch

    a = np.array([[0, 1.2, 0],[2, 3.1, 0],[0.5, 0, 0]])
    idx = a.nonzero() # (row, col)
    data = a[idx]

    # to torch tensor
    idx_t = torch.LongTensor(np.vstack(idx))
    data_t = torch.FloatTensor(data)
    coo_a = torch.sparse_coo_tensor(idx_t, data_t, a.shape)


def load_vocab(vcoab_path):
    '''
    load vocab
    :param vcoab_path:
    :return:
    '''
    vocab = {}
    with open(vcoab_path, "r") as rfp:
        for line in rfp:
            v = line.strip()
            vocab[v] = len(vocab)
    return vocab


def subprocess_popen(statement):
    '''
    execute shell cmd
    :param statement:
    :return:
    '''
    p = subprocess.Popen(statement, shell=True, stdout=subprocess.PIPE)
    while p.poll() is None:
        if p.wait() != 0:
            print("fail.")
            return False
        else:
            re = p.stdout.readlines()
            result = []
            for i in range(len(re)):
                res = re[i].decode('utf-8').strip('\r\n')
                result.append(res)
            return result


# not {'O', 'U', 'Z', 'J', 'B'}
# Common amino acids
common_amino_acid_set = {'R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'}


def clean_seq(protein_id, seq):
    seq = seq.upper()
    new_seq = ""
    has_invalid_char = False
    invalid_char_set = set()
    for ch in seq:
        if 'A' <= ch <= 'Z' and ch not in ['J']:
            new_seq += ch
        else:
            invalid_char_set.add(ch)
            has_invalid_char = True
    if has_invalid_char:
        print("id: %s. Seq: %s" % (protein_id, seq))
        print("invalid char set:", invalid_char_set)
    return new_seq


def load_trained_model(model_config, args, model_class, model_dirpath):
    # load exists checkpoint
    print("load pretrained model: %s" % model_dirpath)
    try:
        model = model_class.from_pretrained(model_dirpath, args=args)
    except Exception as e:
        model = model_class(model_config, args=args)
        pretrained_net_dict = torch.load(os.path.join(args.model_dirpath, "pytorch.pth"),
                                         map_location=torch.device("cpu"))
        model_state_dict_keys = set()
        for key in model.state_dict():
            model_state_dict_keys.add(key)
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            if k.startswith("module."):
                # remove `module.`
                name = k[7:]
            else:
                name = k
            if name in model_state_dict_keys:
                new_state_dict[name] = v
        # print("diff:")
        # print(model_state_dict_keys.difference(new_state_dict.keys()))
        model.load_state_dict(new_state_dict)
    return model

