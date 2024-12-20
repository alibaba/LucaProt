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
@datetime: 2022/12/1 11:20
@project: DeepProtFunc
@file: subword
@desc: xxxx
Reference:
https://github.com/kexinhuang12345/ESPF/blob/master/
'''

import os.path
import codecs
import argparse
from subword_nmt.get_vocab import get_vocab
from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", default="learn_bpe", type=str, required=True, choices=["learn_bpe", "apply_bpe", "learn_joint_bpe_and_vocab", "tokenize", "get_vocab", "subword_vocab_2_token_vocab"],  help="subword running type.")
    '''For Learn and Apply'''
    parser.add_argument("--infile",  type=str, default="../subword/rdrp/all_sequence.txt", help="corpus")
    parser.add_argument("--outfile",  type=str, default="../subword/rdrp/protein_codes_rdrp_1000.txt", help="output filepath")

    '''For Learn'''
    parser.add_argument("--min_frequency",  type=int, default=2, help="min frequency")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--is_dict", action="store_true", help="is dict")
    parser.add_argument("--num_symbols", type=int, default=10000, help="the specified vocab size")
    parser.add_argument("--num_workers", type=int, default=8, help="worker number")

    '''For Tokenize and Apply'''
    parser.add_argument("--codes_file",  type=str, default="../subword/rdrp/protein_codes_rdrp_1000.txt", help="subword codes filepath")
    '''For Tokenize '''
    parser.add_argument("--seq",  type=str, default=None, help="the sequence that want to tokenize")

    run_args = parser.parse_args()
    return run_args


def read_fasta(filepath, exclude):
    '''
    read fasta file
    :param filepath: fasta filepath
    :param exclude: exclude fasta filepath
    :return:
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
                                # print("in exclude list: %s" %uuid)
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
                        # print("in exclude list: %s" %uuid)
        print("%s: %d" %(cur_filepath, total))

    return dataset


def fasta_to_corpus(input_filepath, save_filepath):
    '''
    fasta to sequence corpus
    :param input_filepath:
    :param save_filepath:
    :return:
    '''
    dataset = read_fasta(input_filepath, exclude=None)
    with open(save_filepath, "w") as wfp:
        for item in dataset:
            wfp.write(item[1] + "\n")


def learn(args):
    learn_bpe(
        infile=open(args.infile, "r"),
        outfile=open(args.outfile, "w"),
        min_frequency=args.min_frequency,
        verbose=args.verbose,
        is_dict=args.is_dict,
        num_symbols=args.num_symbols,
        num_workers=args.num_workers
    )


def apply(args):
    bpe_codes_prot = codecs.open(args.codes_file)
    bpe = BPE(codes=bpe_codes_prot)
    bpe.process_lines(
        args.infile,
        open(args.outfile, "w"),
        num_workers=args.num_workers
    )


def vocab(args):
    get_vocab(
        open(args.infile, "r"),
        open(args.outfile, "w")
    )


def subword_vocab_2_token_vocab(args):
    '''
    transform subword results into vocab
    :param args:
    :return:
    '''
    vocabs = set()
    with open(args.infile, "r") as rfp:
        for line in rfp:
            v = line.strip().split()[0].replace("@@", "")
            vocabs.add(v)
    vocabs = ['[PAD]', '[UNK]', '[SEP]', '[CLS]', '[MASK]'] + sorted(list(vocabs), key=lambda x:(len(x), x))
    with open(args.outfile, "w") as wfp:
        for v in vocabs:
            wfp.write(v + "\n")


def tokenize(args):
    bpe_codes_prot = codecs.open(args.codes_file)
    bpe = BPE(bpe_codes_prot, merges=-1, separator='')
    p = bpe.process_line(args.seq).split()
    return p


if __name__ == "__main__":
    input_args = get_args()
    for attr, value in sorted(input_args.__dict__.items()):
        print("\t{}={}".format(attr, value))
    if input_args.func != "tokenize":
        dir_path = os.path.dirname(input_args.outfile)
        if not os.path.exists(dir_path):
            print("Warning: ouput dir %s not exists, created!" % dir_path)
            os.makedirs(dir_path)

    if input_args.func == "learn_bpe":
        #  python subword.py --func learn_bpe --infile ../subword/rdrp/all_sequence.txt --outfile ../subword/rdrp/protein/binary_class/protein_codes_rdrp_1000.txt --verbose
        #  python subword.py --func learn_bpe --infile ../data/rdrp/all_rdrp_domain.fas --outfile ../subword/rdrp/protein/binary_class/all_rdrp_domain_codes_1000.txt --verbose --num_symbols 1000
        #  python subword.py --func learn_bpe --infile ../data/rdrp/all_rdrp_motifABC.fas --outfile ../subword/rdrp/protein/binary_class/all_rdrp_motif_codes_1000.txt --verbose --num_symbols 1000
        #  python subword.py --func learn_bpe --infile ../data/rdrp/RdRp20211115.fasta --outfile ../subword/rdrp/protein/binary_class/all_rdrp_codes_1000.txt --verbose --num_symbols 1000
        if ".fas" in input_args.infile:
            # transform fasta to corpus
            print('fasta convect to corpus txt')
            savepath = os.path.join(os.path.dirname(input_args.infile), ".".join(os.path.basename(input_args.infile).split(".")[0:-1]) + ".txt")
            if os.path.exists(savepath):
                raise Exception("Save path :%s exsits!" % savepath)
            fasta_to_corpus(input_args.infile, savepath)
            input_args.infile = savepath
        learn(input_args)
    elif input_args.func == "tokenize":
        #  python subword.py --func tokenize --seq IPKIDNPEFASQYRPISCCNIFYKCISKMFCSRLKAVVLHLVAENQAAFVQGSQARGGAMDRITTTTRKFE --codes_file ../subword/rdrp/protein_codes_rdrp_1000.txt
        print("input seq:")
        print(input_args.seq)
        print("input seq size:")
        print(len(input_args.seq))
        token = tokenize(input_args)
        print("seq tokenize output:")
        print(token)
        print("seq tokenize size:")
        print(len(token))
    elif input_args.func == "apply_bpe":
        #  python subword.py --func apply_bpe --infile ../subword/rdrp/all_sequence.txt --codes_file ../subword/rdrp/protein/binary_class/protein_codes_rdrp_1000.txt --outfile ../subword/rdrp/protein/binary_class/all_sequence_token_1000.txt
        #  python subword.py --func apply_bpe --infile ../data/rdrp/all_rdrp_domain.txt --codes_file ../subword/rdrp/protein/binary_class/all_rdrp_domain_codes_1000.txt --outfile ../subword/rdrp/protein/binary_class/all_rdrp_domain_token_1000.txt
        #  python subword.py --func apply_bpe --infile ../data/rdrp/all_rdrp_motifABC.txt --codes_file ../subword/rdrp/protein/binary_class/all_rdrp_motif_codes_1000.txt --outfile ../subword/rdrp/protein/binary_class/all_rdrp_motif_token_1000.txt
        #  python subword.py --func apply_bpe --infile ../data/rdrp/RdRp20211115.txt --codes_file ../subword/rdrp/protein/binary_class/all_rdrp_codes_1000.txt --outfile ../subword/rdrp/protein/binary_class/RdRp20211115_token_1000.txt
        if ".fas" in input_args.infile:
            # fasta，not sequence tokenization corpus
            raise Exception("the input file is fasta，not sequence tokenization corpus")
        apply(input_args)
    elif input_args.func == "get_vocab":
        #  python subword.py --func get_vocab --infile ../subword/rdrp/protein/binary_class/all_sequence_token_1000.txt --outfile ../subword/rdrp/protein/binary_class/subword_vocab_1000.txt
        #  python subword.py --func get_vocab --infile ../subword/rdrp/protein/binary_class/all_rdrp_domain_token_1000.txt --outfile ../subword/rdrp/protein/binary_class/all_rdrp_domain_vocab_1000.txt
        #  python subword.py --func get_vocab --infile ../subword/rdrp/protein/binary_class/all_rdrp_motif_token_1000.txt --outfile ../subword/rdrp/protein/binary_class/all_rdrp_motif_vocab_1000.txt
        #  python subword.py --func get_vocab --infile ../subword/rdrp/protein/binary_class/RdRp20211115_token_1000.txt --outfile ../subword/rdrp/protein/binary_class/RdRp20211115_vocab_1000.txt
        vocab(input_args)
    elif input_args.func == "subword_vocab_2_token_vocab":
        #  python subword.py --func subword_vocab_2_token_vocab --infile ../subword/rdrp/protein/binary_class/subword_vocab_1000.tx --outfile ../vocab/rdrp/protein/binary_class/subword_vocab_1000.txt
        #  python subword.py --func subword_vocab_2_token_vocab --infile ../subword/rdrp/protein/binary_class/all_rdrp_domain_vocab_1000.txt --outfile ../vocab/rdrp/aprotein/binary_class/ll_rdrp_domain_vocab_1000.txt
        #  python subword.py --func subword_vocab_2_token_vocab --infile ../subword/rdrp/protein/binary_class/all_rdrp_motif_vocab_1000.txt --outfile ../vocab/rdrp/protein/binary_class/all_rdrp_motif_vocab_1000.txt
        #  python subword.py --func subword_vocab_2_token_vocab --infile ../subword/rdrp/protein/binary_class/RdRp20211115_vocab_1000.txt --outfile ../vocab/rdrp/protein/binary_class/RdRp20211115_vocab_1000.txt
        subword_vocab_2_token_vocab(input_args)