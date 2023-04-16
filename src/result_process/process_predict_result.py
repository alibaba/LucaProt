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
@datetime: 2023/1/18 14:03
@project: DeepProtFunc
@file: process_predict_result.py
@desc: process the results of prediction
'''
import os, csv
import io, textwrap, itertools
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


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
    :param filepath: the save filepath
    :param sequences: fasta sequence list (each item: [id, seq])
    :return:
    '''
    with open(filepath, "w") as output_handle:
        for sequence in sequences:
            SeqIO.write(sequence, output_handle, "fasta")


def main(input_fasta_filepaths, result_filedir, result_filenames, merge_dirname):
    if isinstance(input_fasta_filepaths, str):
        input_fasta_filepaths = [input_fasta_filepaths]
    total = 0
    fasta_id_set = set()
    repeat_cnt = 0
    for input_fasta_filepath in input_fasta_filepaths:
        cnt = 0
        for row in fasta_reader(input_fasta_filepath):
            cnt += 1
            if row[0] in fasta_id_set:
                repeat_cnt += 1
            fasta_id_set.add(row[0])
        total += cnt
    print("total: %d, protein id: %d, repeat cnt: %d" % (total, len(fasta_id_set), repeat_cnt))
    # assert len(fasta_id_set) == total
    print("fasta num: %d" % total)
    writer_dir = os.path.join(result_filedir, merge_dirname)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    all_result_wfp = open(os.path.join(writer_dir, "predict_all.csv"), "w")
    all_result_wtiter = csv.writer(all_result_wfp)
    all_result_wtiter.writerow(["id", "prob", "pred"])
    positive_result_wfp = open(os.path.join(writer_dir, "predict_positive.csv"), "w")
    positive_result_writer = csv.writer(positive_result_wfp)
    positive_result_writer.writerow(["id", "prob", "pred"])
    sequence_list = []
    total = 0
    pred_fasta_id = {}
    for filename in result_filenames:
        filepath = os.path.join(result_filedir, filename)
        with open(filepath, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1:
                    continue
                protein_id, seq, predict_prob, predict_label, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
                if protein_id in pred_fasta_id:
                    pred_fasta_id[protein_id].append((predict_prob, predict_label))
                else:
                    pred_fasta_id[protein_id] = [(predict_prob, predict_label)]
                fasta_id_set.remove(protein_id)

                protein_id = protein_id[1:] if protein_id and protein_id[0] == ">" else protein_id
                if int(predict_label) == 1:
                    sequence_list.append(SeqRecord(
                        Seq(seq, None),
                        id=protein_id,
                        description=""))
                    positive_result_writer.writerow([protein_id, predict_prob, predict_label])
                all_result_wtiter.writerow([protein_id, predict_prob, predict_label])
            total += cnt - 1
    print("predict num: %d" % total)
    if len(fasta_id_set) > 0:
        print("not done set:")
        print(fasta_id_set)
    pred_fasta_id = [item for item in pred_fasta_id.items() if len(item[1]) > 1]
    if len(pred_fasta_id) > 0:
        print("done >= 2 times set:")
        print(pred_fasta_id)
    print("total: %d, positive: %d, p rate: %f" % (total, len(sequence_list), len(sequence_list)/total))
    write_fasta(os.path.join(writer_dir, "predict_positive.fasta"), sequence_list)
    all_result_wfp.close()
    positive_result_wfp.close()


if __name__ == "__main__":
    '''
    input_fasta_filepath = "/mnt/****/biodata/20230108-to-Ali/00self_sequecing_500aa.pep"
    result_filedir = "../predicts/rdrp_40/protein/binary_class/sefn/20230107005818/"
    result_filenames = ["00self_sequecing_500aa_001_with_pdb_emb/pred_result.csv",
                        "00self_sequecing_500aa_002_with_pdb_emb/pred_result.csv",
                        "00self_sequecing_500aa_003_with_pdb_emb/pred_result.csv"]
    merge_dirname = "00self_sequecing_500aa"
    main(input_fasta_filepath, result_filedir, result_filenames, merge_dirname)
    '''
    input_fasta_filepath = ["/mnt2/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_001.pep",
                            "/mnt2/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_002.pep",
                            "/mnt2/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_003.pep",
                            "/mnt2/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_004.pep",
                            "/mnt2/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_005.pep",
                            "/mnt2/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_006.pep"]
    result_filedir = "../predicts/rdrp_40/protein/binary_class/sefn/20230107005818/checkpoint-95000/"
    result_filenames = ["all_500aa.part_003_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_004_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_005_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_006_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_001_001_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_001_002_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_001_003_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_001_004_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_002_001_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_002_002_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_002_003_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_002_004_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_003_001_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_003_002_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_003_003_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_003_004_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_004_001_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_004_002_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_004_003_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_004_004_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_005_001_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_005_002_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_005_003_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_005_004_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_006_001_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_006_002_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_006_003_with_pdb_emb/pred_result.csv",
                        "all_500aa.part_006_004_with_pdb_emb/pred_result.csv"
                        ]
    merge_dirname = "all_500aa.pep.split"
    main(input_fasta_filepath, result_filedir, result_filenames, merge_dirname)