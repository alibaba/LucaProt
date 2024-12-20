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
@datetime: 2022/11/27 14:45
@project: DeepProtFunc
@file: data_preprocess_for_rdrp
@desc: construct a dataset for model building, consists of virus RdRP and non-virus RdRP
'''
import csv
import os, sys
import random
import requests
import codecs
import shutil
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from subword_nmt.get_vocab import get_vocab
from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import plot_bins, write_fasta, file_reader, subprocess_popen
except ImportError:
    from src.utils import plot_bins, write_fasta, file_reader, subprocess_popen


def k_mer(seq, k=1):
    '''
    k-mer
    :param seq:
    :param k:
    :return:
    '''
    k_mer_set = set()
    for i in range(len(seq) - k + 1):
        k_mer_set.add(seq[i: i + k])
    return k_mer_set


def generate_vocab_file(
        dataset_filepath,
        save_filepath,
        k=1
):
    '''
    building vocab
    :param dataset_filepath:
    :param save_filepath:
    :param k: k_mer size
    :return:
    '''
    vocabs = set()
    if isinstance(dataset_filepath, str):
        with open(dataset_filepath, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1:
                    continue
                seq = row[1]
                if k == 1:
                    vocabs = vocabs.union(set(seq))
                else:
                    vocabs = vocabs.union(k_mer(seq, k=k))
    elif isinstance(dataset_filepath, list):
        dataset = dataset_filepath
        for row in dataset:
            seq = row[1]
            if k == 1:
                vocabs = vocabs.union(set(seq))
            else:
                vocabs = vocabs.union(k_mer(seq, k=k))

    vocabs = ['[PAD]', '[UNK]', '[SEP]', '[CLS]', '[MASK]'] + list(vocabs)
    with open(save_filepath, "w") as wfp:
        for v in vocabs:
            wfp.write(v + "\n")


def generate_label_file(save_filepath):
    '''
    building label list file
    :param save_filepath:
    :return:
    '''
    with open(save_filepath, "w") as wfp:
        wfp.write("label\n")
        wfp.write("0\n")
        wfp.write("1\n")


def read_fasta(filepath, exclude):
    '''
    read sequences of fasta file
    :param filepath: fasta filepath
    :param exclude: fasta filepath of exclude sequences
    :return: dataset
    '''
    exclude_ids = set()
    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        for p in exclude:
            with open(p, "r") as rfp:
                for line in rfp:
                    protein_id = line.strip().split("|")[1]
                    exclude_ids.add(protein_id)

    if isinstance(filepath, str):
        filepath = [filepath]
    dataset = []
    for cur_filepath in filepath:
        total = 0
        with open(cur_filepath, "r") as rfp:
            seq = ""
            uuid = ""
            for line in rfp:
                line = line.strip()
                if line.startswith(">"):
                    if seq and len(seq) > 0:
                        if len(exclude_ids) == 0:
                            dataset.append([uuid, seq])
                            total += 1
                        else:
                            strs = uuid.strip().split("|")
                            if len(strs) <= 1 or strs[1] not in exclude_ids:
                                dataset.append([uuid, seq])
                                total += 1
                            else:
                                pass
                    uuid = line
                    seq = ""
                else:
                    seq += line
            if seq and uuid and len(seq) > 0:
                if len(exclude_ids) == 0:
                    dataset.append([uuid, seq])
                    total += 1
                else:
                    strs = uuid.strip().split("|")
                    if len(strs) <= 1 or strs[1] not in exclude_ids:
                        dataset.append([uuid, seq])
                        total += 1
                    else:
                        pass
        print("%s: %d" % (cur_filepath, total))

    return dataset


def select_sequence_from_uniprot(
        protein_ids,
        filepaths,
        save_filepath,
        not_found_save_filepath
):
    '''
    get sequences by id from local uniprot files
    :param protein_ids: id lists
    :param filepaths: uniprot filepaths
    :param save_filepath: sequences
    :param not_found_save_filepath: unfounded id list saved filepath
    :return:
    '''
    found_data = {}
    found_protein_ids = set()
    if isinstance(protein_ids, list):
        protein_ids = set(protein_ids)
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for filepath in filepaths:
        with open(filepath, "r") as rfp:
            seq = []
            proten_id = ""
            uuid = ""
            for line in rfp:
                line = line.strip()
                if line.startswith(">"):
                    if seq and len(seq) > 0:
                        if proten_id in protein_ids:
                            found_data[uuid] = "\n".join(seq)
                            found_protein_ids.add(proten_id)
                    proten_id = line.split("|")[1]
                    uuid = line
                    seq = []
                else:
                    seq.append(line)
            if proten_id and seq and len(seq) > 0:
                if proten_id in protein_ids:
                    found_data[uuid] = "\n".join(seq)
                    found_protein_ids.add(proten_id)
    not_found_protein_ids = protein_ids.difference(found_protein_ids)
    if save_filepath:
        with open(save_filepath, "w") as wfp:
            for item in found_data.items():
                wfp.write(item[0] + "\n")
                wfp.write(item[1] + "\n")
        with open(not_found_save_filepath, "w") as wfp:
            for protein_id in not_found_protein_ids:
                wfp.write(protein_id + "\n")
    return found_data, not_found_protein_ids


def write_fasta_2_csv(
        dataset,
        filepath,
        mode,
        source,
        label=None,
        sampler_num=None,
        min_length=None
):
    '''
    write dataset to file
    :param dataset:
    :param filepath:
    :param mode:
    :param source:
    :param label:
    :param sampler_num:
    :param min_length:
    :return:
    '''
    print("dataset size: %d" %len(dataset))

    removed_dataset = []
    selected_dataset = dataset
    if sampler_num and len(dataset) > sampler_num:
        if min_length:
            dataset = [v for v in dataset if len(v[1]) >= min_length]
        print("dataset size: %d" %len(dataset))
        for _ in range(5):
            random.shuffle(dataset)
        if len(dataset) > sampler_num:
            removed_dataset = dataset[sampler_num:]
            dataset = dataset[0:sampler_num]
        selected_dataset = dataset

    print("dataset size: %d" %len(dataset))
    with open(filepath, mode) as wfp:
        writer = csv.writer(wfp)
        if mode == "w":
            writer.writerow(["id", "seq", "label", "source"])
        for item in dataset:
            if label is not None:
                item.append(label)
            item.append(source)
            writer.writerow(item)
    return selected_dataset, removed_dataset


def split_dataset(
        filepath,
        save_dir,
        rate=0.7
):
    '''
    The dataset is randomly divided into training set/validation set/test set according to the specified ratio
    :param filepath:
    :param save_dir:
    :param rate:
    :return:
    '''
    dataset = {}
    if isinstance(filepath, str):
        filepath = [filepath]
    for cur_filepath in filepath:
        with open(cur_filepath, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1:
                    continue
                label = row[2]
                source = row[3]
                if label not in dataset:
                    dataset[label] = {}
                    dataset[label][source] = []
                elif source not in dataset[label]:
                    dataset[label][source] = []
                dataset[label][source] .append(row)
    train_set = []
    dev_set = []
    test_set = []
    for item1 in dataset.items():
        for item2 in item1[1].items():
            cur_label_source_dataset = item2[1]
            for _ in range(5):
                random.shuffle(cur_label_source_dataset)
            size1 = int(len(cur_label_source_dataset) * rate)
            train_set.extend(cur_label_source_dataset[0: size1])
            size2 = int(len(cur_label_source_dataset) * (0.5 + rate/2))
            dev_set.extend(cur_label_source_dataset[size1: size2])
            test_set.extend(cur_label_source_dataset[size2:])
    dataset_list = [train_set, dev_set, test_set]
    filename_list = ["train.csv", "dev.csv", "test.csv"]
    for idx, dataset in enumerate(dataset_list):
        with open(os.path.join(save_dir, filename_list[idx]), "w") as wfp:
            writer = csv.writer(wfp)
            writer.writerow(["id", "seq", "label", "source"])
            for item in dataset:
                writer.writerow(item)


def generate_sequence_corpus(dataset, save_filepath):
    '''
    write all sequence into a filepath, which is to extract subword vocab
    :param dataset:
    :param save_filepath:
    :return:
    '''
    with open(save_filepath, "w") as wfp:
        for item in dataset:
            wfp.write(item[1] + "\n")


def dataset_stats(dataset):
    '''
    statistic the sequence length distribution, and plot it
    :param dataset: the dataset
    :return:
    '''
    size_stats = {}
    avg_len = 0
    size_list = []
    for item in dataset:
        size = len(item[1])
        if size not in size_stats:
            size_stats[size] = 1
        else:
            size_stats[size] += 1
        avg_len += size
        size_list.append(size)
    size_stats = sorted(size_stats.items(), key=lambda x:x[0], reverse=True)
    print("dataset size: ", len(dataset))
    print("dataset max seq length:", size_stats[0][0])
    print("dataset min seq length:", size_stats[-1][0])
    print("dataset avg seq length:", avg_len//len(dataset))
    return size_list


def subword_vocab_2_token_vocab(subword_vocab_filepath, token_vocab_filepath):
    '''
    extract the subword vocab
    :param subword_vocab_filepath:
    :param token_vocab_filepath:
    :return:
    '''
    vocabs = set()
    with open(subword_vocab_filepath, "r") as rfp:
        for line in rfp:
            v = line.strip().split()[0].replace("@@", "")
            vocabs.add(v)
    vocabs = ['[PAD]', '[UNK]', '[SEP]', '[CLS]', '[MASK]'] + sorted(list(vocabs), key=lambda x:(len(x), x))
    with open(token_vocab_filepath, "w") as wfp:
        for v in vocabs:
            wfp.write(v + "\n")


fasta_url = 'https://www.uniprot.org/uniprot/'


def get_sequence_from_api(filepath, save_filepath, err_filepath):
    '''
    get sequence from uniprot by API
    :param filepath:
    :param save_filepath:
    :param err_filepath:
    :return:
    '''
    err_fp = open(err_filepath, "w")
    err_cnt = 0
    suc_cnt = 0
    with open(save_filepath, "w") as wfp:
        with open(filepath, "r") as rfp:
            cnt = 0
            for line in rfp:
                cnt += 1
                if cnt == 1:
                    continue
                strs = line.strip().split("\t")
                uniprot_id = strs[-1].split(".")[0]
                try:
                    fasta_URL = fasta_url + uniprot_id + '.fasta'
                    request = requests.post(fasta_URL)
                    request.raise_for_status()
                    fasta_string = request.text.split('\n')
                    info = fasta_string[0]
                    if len(info) == 0 or len(fasta_string) < 2:
                        err_cnt += 1
                        err_fp.write(uniprot_id + "\n")
                        continue
                    sequence = '\n'.join(fasta_string[1:])
                    wfp.write(info + "\n")
                    wfp.write(sequence)
                    suc_cnt += 1
                    if (cnt - 1) % 100 == 0:
                        # time.sleep(random.randint(1, 60))
                        wfp.flush()
                        err_fp.flush()
                        print("had done %d, err: %d" % (suc_cnt, err_cnt))
                except ImportError:
                    err_cnt += 1
                    err_fp.write(uniprot_id + "\n")
    print("done, %d, err: %d" % (suc_cnt, err_cnt))
    err_fp.close()


def get_sequence_from_local_db(filepath, save_filepath, lib_filepaths=None):
    '''
    get the sequence from local uniprot files
    :param filepath:
    :param save_filepath:
    :param lib_filepaths:
    :return:
    '''
    # ori_uniprot_dir = "/bio/uniprot/knowledgebase/"
    if lib_filepaths is None:
        ori_uniprot_dir = "/mnt2/****/biodata/uniprot/"
        lib_filepaths = [os.path.join(ori_uniprot_dir, "uniprot_trembl.fasta"), os.path.join(ori_uniprot_dir, "uniprot_sprot.fasta")]

    protein_ids = set()
    with open(filepath, "r") as rfp:
        cnt = 0
        for line in rfp:
            cnt += 1
            if cnt == 1:
                continue
            strs = line.strip().split("\t")
            uniprot_id = strs[-1].split(".")[0]
            protein_ids.add(uniprot_id)
    not_found_filepath = os.path.join(os.path.dirname(save_filepath), ".".join(os.path.basename(save_filepath).split(".")[:-1]) + "_not_found_protein_ids.txt")
    found_data, not_found_protein_ids = select_sequence_from_uniprot(protein_ids, lib_filepaths, save_filepath, not_found_filepath)
    print("%s: total: %d, found: %d, not unfound: %d" % (filepath, len(found_data) + len(not_found_protein_ids), len(found_data), len(not_found_protein_ids)))


def load_non_rdrp_from_api():
    '''
    Obtain the complete sequence through the api according to the id of uniprot, because the sequence of the negative sample in rdrp is not a complete protein sequence
    :return:
    '''
    get_sequence_from_api(
        "../../../biodata/20221204-to-Ali/other_virus_pro.info.txt",
        "../data/rdrp/other_virus_uniprot.fasta",
        "../data/rdrp/err_other_virus_pro.txt"
    )
    print("done other_virus_pro")
    get_sequence_from_api(
        "../../../biodata/20221204-to-Ali/non-virus.info.txt",
        "../data/rdrp/non_virus_uniprot.fasta",
        "../data/rdrp/err_non_virus.txt"
    )
    print("done non virus")


def load_non_rdrp_from_local_db():
    '''
    Obtain the complete sequence of negative samples through the sequence library of uniprot local files
    :return:
    '''
    # ori_uniprot_dir = "/bio/uniprot/knowledgebase/"
    ori_uniprot_dir = "/mnt2/****/biodata/uniprot/"
    filepaths = [os.path.join(ori_uniprot_dir, "uniprot_trembl.fasta"), os.path.join(ori_uniprot_dir, "uniprot_sprot.fasta")]

    # non-rdrp of viral proteins
    protein_ids = set()
    with open("../../../biodata/20221204-to-Ali/other_virus_pro.info.txt", "r") as rfp:
        cnt = 0
        for line in rfp:
            cnt += 1
            if cnt == 1:
                continue
            strs = line.strip().split("\t")
            uniprot_id = strs[-1].split(".")[0]
            protein_ids.add(uniprot_id)
    save_filepath = "../data/rdrp/other_virus_pro_sequence.fasta"
    not_found_filepath = "../data/rdrp/other_virus_not_found_protein_ids.txt"
    found_data, not_found_protein_ids = select_sequence_from_uniprot(protein_ids, filepaths, save_filepath, not_found_filepath)
    print("other virus: total: %d, found: %d, not unfound: %d" %(len(found_data) + len(not_found_protein_ids),
                                                                 len(found_data), len(not_found_protein_ids)))
    # non-viral protein
    protein_ids = set()
    with open("../../../biodata/20221204-to-Ali/non-virus.info.txt", "r") as rfp:
        cnt = 0
        for line in rfp:
            cnt += 1
            if cnt == 1:
                continue
            strs = line.strip().split("\t")
            uniprot_id = strs[-1].split(".")[0]
            protein_ids.add(uniprot_id)
    save_filepath = "../data/rdrp/non_virus_sequence.fasta"
    not_found_filepath = "../data/rdrp/non_virus_not_found_protein_ids.txt"
    found_data, not_found_protein_ids = select_sequence_from_uniprot(protein_ids, filepaths, save_filepath, not_found_filepath)
    print("non virus: total: %d, found: %d, not unfound: %d" %(
        len(found_data) + len(not_found_protein_ids),
        len(found_data),
        len(not_found_protein_ids)
    ))


def preparing_file_for_prediction(
        filepath_list,
        save_path,
        ground_truth_col_index_list=None,
        label_list=None
):
    '''
    prepare file for prediction
    :param filepath_list: the input data filepath list, original file list
    :param save_path: prepared file savepath
    :param ground_truth_col_index_list:
    :param label_list:
    If ground_truth_col_index is null or -1, it means that there is no ground truth column in each file in the filepath_list, otherwise it means that there is a ground truth column in the original file.
    If the label_list is not null and not empty, it means that the original file has ground truth, and each file has the same label, corresponding to label_list index.
    If both ground_truth_col_index and label_list are null, it means that the dataset to be predicted has no ground truth.
    :return:
    '''
    if ground_truth_col_index_list:
        assert len(ground_truth_col_index_list) == len(filepath_list)
    if label_list:
        assert len(label_list) == len(filepath_list)
    assert save_path is not None and not os.path.exists(save_path)
    assert os.path.exists(os.path.dirname(save_path))
    all_dataset = []
    for idx, filepath in enumerate(filepath_list):
        if ".fas" in filepath:
            dataset = read_fasta(filepath, exclude=None)
            if label_list:
                label = label_list[idx]
            else:
                label = None
            for item in dataset:
                item.append(label)
                all_dataset.append(item)
        elif ".csv" in filepath:
            if label_list:
                label = label_list[idx]
            else:
                label = None
            with open(filepath, "r") as rfp:
                reader = csv.reader(rfp)
                cnt = 0
                for row in reader:
                    cnt += 1
                    if cnt == 1:
                        continue
                    uuid, seq = row[0], row[1]
                    if ground_truth_col_index_list:
                        label = row[ground_truth_col_index_list[idx]]
                    all_dataset.append([uuid, seq, label])

    with open(save_path, "w") as wfp:
        writer = csv.writer(wfp)
        writer.writerow(["id", "seq", "label"])
        for item in all_dataset:
            writer.writerow(item)


# prepare the dataset
if __name__ == "__main__":
    '''
    # btain the complete sequence through the api according to the uniprot id, because the sequence of the negative sample in rdrp dataset is not a complete protein sequence
    load_non_rdrp_from_api()
    
    # Obtain the complete sequence of negative samples in rdrp dataset through the local library of uniprot
    load_non_rdrp_from_local_db()
    print("negative sequence get done.")
    '''

    '''
    positive samples in rdrp dataset
    '''
    # the sampling multiple relative to the positive sample
    times = 40
    dataset_name = "rdrp_%d" %times
    # read raw data(positive)
    positive_dataset = read_fasta("../data/rdrp/RdRp20211115.fasta", exclude=None)
    # statistics
    size_list = dataset_stats(positive_dataset)
    # plot the sequence length distribution
    pic_save_filepath = "../pics/%s/rdrp_sequence_length_distribution.png" %dataset_name
    if not os.path.exists("../pics/%s/" % dataset_name):
        os.makedirs("../pics/%s/" % dataset_name)
    plot_bins(size_list, xlabel="sequence length", ylabel="distribution", bins=40, filepath=pic_save_filepath)
    # write the positive samples into file
    if not os.path.exists("../dataset/%s/protein/binary_class/" % dataset_name):
        os.makedirs("../dataset/%s/protein/binary_class/" % dataset_name)
    positive_selected_dataset, positive_removed_dataset = write_fasta_2_csv(
        positive_dataset,
        "../dataset/%s/protein/binary_class/rdrp_dataset.csv" % dataset_name,
        mode='w',
        source="rdrp",
        label=1
    )
    print("rdrp positive dataset generate done.")

    '''
    other proteins of virus(negative samples)
    '''
    # read raw data(negative)
    negative_dataset_1 = read_fasta(
        ["../data/rdrp/other_virus_pro_sequence.fasta"],
        exclude=["../data/rdrp/other_virus_pro_sequence_exclude.fasta"]
    )
    # statistics
    size_list = dataset_stats(negative_dataset_1)
    # plot sequence length distribution
    pic_save_filepath = "../pics/%s/other_virus_sequence_length_distribution.png" % dataset_name
    plot_bins(size_list, xlabel="sequence length", ylabel="distribution", bins=40, filepath=pic_save_filepath)
    # write the negative samples into file
    # negative samples
    negative_selected_dataset_1, negative_removed_dataset_1 = write_fasta_2_csv(
        negative_dataset_1,
        "../dataset/%s/protein/binary_class/other_virus_dataset.csv" % dataset_name,
        mode='w',
        source="other_virus",
        label=0,
        sampler_num=None,
        min_length=None
    )
    print("other virus negative dataset generate done.")


    '''
    the domains of other proteins of virus(negative samples)
    '''
    # read raw data(negative)
    negative_dataset_2 = read_fasta(["../data/rdrp/other_virus_pro_domain_sequence.fasta"], exclude=None)
    # statistics
    size_list = dataset_stats(negative_dataset_2)
    # plot sequence length distribution
    pic_save_filepath = "../pics/%s/other_virus_domain_sequence_length_distribution.png" % dataset_name
    plot_bins(size_list, xlabel="sequence length", ylabel="distribution", bins=40, filepath=pic_save_filepath)
    # write the negative samples into file
    # negative samples
    negative_selected_dataset_2, negative_removed_dataset_2 = write_fasta_2_csv(
        negative_dataset_2,
        "../dataset/%s/protein/binary_class/other_virus_domain_dataset.csv" % dataset_name,
        mode='w',
        source="other_virus_domain",
        label=0,
        sampler_num=None,
        min_length=None
    )
    print("other virus domain negative dataset generate done.")

    '''
    other proteins of non virus(negative samples)
    '''
    # read raw data(negative)
    negative_dataset_3 = read_fasta(
        ["../data/rdrp/non_virus_sequence.fasta"],
        exclude=["../data/rdrp/non_virus_sequence_exclude.fasta"]
    )
    # statistics
    size_list = dataset_stats(negative_dataset_3)
    # plot sequence length distribution
    pic_save_filepath = "../pics/%s/non_virus_sequence_length_distribution.png" % dataset_name
    plot_bins(size_list, xlabel="sequence length", ylabel="distribution", bins=40, filepath=pic_save_filepath)
    # write the negative samples into file
    # randomly sampling: N times positive sample dataset，Filter out those with sequence length less than 100
    negative_selected_dataset_3, negative_removed_dataset_3 = write_fasta_2_csv(
        negative_dataset_3,
        "../dataset/%s/protein/binary_class/non_virus_dataset.csv" % dataset_name,
        mode='w',
        source="non_virus",
        label=0,
        sampler_num=times * len(positive_dataset),
        min_length=100
    )
    print("non virus negative dataset generate done.")

    # generate label file
    generate_label_file("../dataset/%s/protein/binary_class/label.txt" % dataset_name)
    print("label file generate done.")

    dataset = negative_dataset_1 + negative_dataset_2 + negative_dataset_3 + positive_dataset
    print("total dataset size: %d" % len(dataset))
    selected_dataset = positive_selected_dataset + negative_selected_dataset_1 + negative_selected_dataset_2 + negative_selected_dataset_3 + negative_removed_dataset_3[0:50000]
    print("total selected dataset size: %d" % len(selected_dataset))
    # generate char-level vocab file
    if not os.path.exists("../vocab/%s/protein/binary_class" % dataset_name):
        os.makedirs("../vocab/%s/protein/binary_class" % dataset_name)
    generate_vocab_file(dataset, save_filepath="../vocab/%s/protein/binary_class/vocab.txt" % dataset_name)
    print("char-level vocabulary generate done.")

    # Randomly divide into training, validation, testing sets
    split_dataset(
        [
            "../dataset/%s/protein/binary_class/rdrp_dataset.csv" % dataset_name,
            "../dataset/%s/protein/binary_class/other_virus_dataset.csv" % dataset_name,
            "../dataset/%s/protein/binary_class/other_virus_domain_dataset.csv" % dataset_name,
            "../dataset/%s/protein/binary_class/non_virus_dataset.csv" % dataset_name
        ],
        "../dataset/%s/protein/binary_class/" % dataset_name,
        rate=0.8
    )
    print("split dataset to train, dev and test done.")

    # generate subword corpus
    if not os.path.exists("../subword/%s/protein/binary_class/" % dataset_name):
        os.makedirs("../subword/%s/protein/binary_class/" % dataset_name)
    generate_sequence_corpus(
        selected_dataset,
        save_filepath="../subword/%s/protein/binary_class/all_sequence.txt" % dataset_name
    )
    print("sequence corpus generate done.")

    # run subword algorithms
    num_workers = 4
    # the vocab size of subword
    subword_num_list = [1000, 2000, 5000, 10000, 20000]
    for subword_num in subword_num_list:
        learn_bpe(
            infile=open("../subword/%s/protein/binary_class/all_sequence.txt" % dataset_name, "r"),
            outfile=open("../subword/%s/protein/binary_class/protein_codes_rdrp_%d.txt" % (dataset_name, subword_num), "w"),
            min_frequency=2,
            verbose=True,
            is_dict=False,
            num_symbols=subword_num,
            num_workers=num_workers
        )
        print("subword for size=%d train done." % subword_num)

        # apply subword results
        if not os.path.exists("../vocab/%s/protein/binary_class/" % dataset_name):
            os.makedirs("../vocab/%s/protein/binary_class/" % dataset_name)
        shutil.copyfile(
            "../subword/%s/protein/binary_class/protein_codes_rdrp_%d.txt" % (dataset_name, subword_num),
            "../vocab/%s/protein/binary_class/protein_codes_rdrp_%d.txt" % (dataset_name, subword_num)
        )
        bpe_codes_prot = codecs.open(
            "../subword/%s/protein/binary_class/protein_codes_rdrp_%d.txt" % (dataset_name, subword_num)
        )
        bpe = BPE(codes=bpe_codes_prot)
        bpe.process_lines(
            "../subword/%s/protein/binary_class/all_sequence.txt" % dataset_name,
            open("../subword/%s/protein/binary_class/all_sequence_token_%d.txt" % (dataset_name, subword_num), "w"),
            num_workers=num_workers
        )
        print("subword for size=%d apply done." % subword_num)

        # generate subword vocabulary
        get_vocab(
            open("../subword/%s/protein/binary_class/all_sequence_token_%d.txt" % (dataset_name, subword_num), "r"),
            open("../subword/%s/protein/binary_class/subword_vocab_%d.txt" % (dataset_name, subword_num), "w")
        )
        if not os.path.exists("../vocab/%s/protein/binary_class/" % dataset_name):
            os.makedirs("../vocab/%s/protein/binary_class/" % dataset_name)
        subword_vocab_2_token_vocab(
            "../subword/%s/protein/binary_class/subword_vocab_%d.txt" % (dataset_name, subword_num),
            "../vocab/%s/protein/binary_class/subword_vocab_%d.txt" % (dataset_name, subword_num)
        )
        print("subword-level for size=%d vocabulary done." % subword_num)


# After the completion of model training, the preparation of prediction data
if __name__ == "__main__1":
    preparing_file_for_prediction(
        ["../data/rdrp/test_data/non-rdrp.fas", "../data/rdrp/test_data/rdrp.fas"],
        save_path="../data/rdrp/test_data/prediction_dataset_with_label.csv",
        label_list=["0", "1"]
    )
    preparing_file_for_prediction(
        ["../data/rdrp/test_data/all_novel_rdrp.fas"],
        save_path="../data/rdrp/test_data/prediction_dataset_only_positive.csv",
        label_list=["1"]
    )
    preparing_file_for_prediction(
        ["../data/rdrp/test_data/uncertain.fas"],
        save_path="../data/rdrp/test_data/prediction_dataset_wo_label.csv",
        label_list=None
    )

# Gets the full sequence according to the id of uniprot
if __name__ == "__main__2":
    filepath = "../data/rdrp/RT.info.txt"
    save_filepath = "../data/rdrp/RT.info_api.fasta"
    err_filepath = "../data/rdrp/RT.info_api_not_found.txt"
    get_sequence_from_api(filepath, save_filepath, err_filepath)

# Gets the sequence of RT(negative) from the local database
if __name__ == "__main__3":
    filepath = "../data/rdrp/RT.info.txt"
    save_filepath = "../data/rdrp/RT.info_local.fasta"
    get_sequence_from_local_db(filepath, save_filepath, lib_filepaths=None)

# Gets the sequence of RT(negative) from the API
if __name__ == "__main__4":
    filepath = "../data/rdrp/RT.info_local_not_found_protein_ids.txt"
    save_filepath = "../data/rdrp/RT.info_local_not_found_protein_ids_api_found.fasta"
    err_filepath = "../data/rdrp/RT.info_local_not_found_protein_ids_api_not_found.txt"
    get_sequence_from_api(filepath, save_filepath, err_filepath)