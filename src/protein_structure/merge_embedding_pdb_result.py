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
@datetime: 2022/12/27 15:58
@project: DeepProtFunc
@file: merge_embedding_pdb_result.py
@desc: merge protein sequence、pdb filepath、embedding filepath info
'''
import os, csv, sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import write_fasta, fasta_reader, csv_reader, txt_reader
except ImportError:
    from src.utils import write_fasta, fasta_reader, csv_reader, txt_reader


def load_file(filepath: str):
    if filepath.endswith(".csv"):
        reader = csv_reader(filepath, header_filter=True, header=True)
    elif filepath.endswith(".txt"):
        reader = txt_reader(filepath, header_filter=False, header=False)
    else:
        reader = fasta_reader(filepath)
    # protein list
    prot_id_list = set()
    prot_id_2_seq = {}
    for row in reader:
        protein_id = row[0].strip()
        seq = row[1].strip()
        prot_id_list.add(protein_id)
        prot_id_2_seq[protein_id] = seq
    return prot_id_list, prot_id_2_seq


def load_pdb(protein_id_2_idx_fileptah, pdb_dirpath):
    '''
    load all PDB files
    :param protein_id_2_idx_fileptah: mapping betwwen protein id and index filepath
    :param pdb_dirpath: pdb saved dir
    :return:
    '''
    if not os.path.exists(pdb_dirpath):
        print("pdb dir: %s not exists!" % pdb_dirpath)
        return None
    protein_2_pdb_filepath = os.path.join(os.path.dirname(os.path.dirname(protein_id_2_idx_fileptah)),
                                          ".".join(os.path.basename(protein_id_2_idx_fileptah).split(".")[0:-1]) + "_protein_2_pdb.csv")
    if os.path.exists(protein_2_pdb_filepath):
        raise Exception("file: %s exists!" % protein_2_pdb_filepath)
    prot_id_list = set()
    done_prot_id_list = set()
    with open(protein_2_pdb_filepath, "w") as wfp:
        writer = csv.writer(wfp)
        writer.writerow(["prot_id", "pdb_filename", "ptm", "mean_plddt"])
        with open(protein_id_2_idx_fileptah, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1:
                    continue
                index, prot_id, seq_len, ptm, mean_plddt = row
                prot_id_list.add(prot_id)
                pdb_path = os.path.join(pdb_dirpath, "protein_%s.pdb" % index)
                if os.path.exists(pdb_path):
                    writer.writerow([prot_id,  "protein_%s.pdb" % index, ptm, mean_plddt])
                    done_prot_id_list.add(prot_id)
                else:
                    writer.writerow([prot_id,  None, ptm, mean_plddt])
    print("pdb want to do: %d, done: %d, undo: %d" % (len(prot_id_list), len(done_prot_id_list), len(prot_id_list.difference(done_prot_id_list))))
    return protein_2_pdb_filepath


def load_emb(protein_id_2_idx_filepath, embedding_dirpath):
    if not os.path.exists(embedding_dirpath):
        raise Exception("emb dir: %s not exists!" % embedding_dirpath)
    protein_2_embedding_filepath = os.path.join(os.path.dirname(protein_id_2_idx_filepath[-1]),
                                                ".".join(os.path.basename(protein_id_2_idx_filepath[-1]).split(".")[0:-1]) + "_protein_2_emb.csv")
    if os.path.exists(protein_2_embedding_filepath):
        raise Exception("file: %s exists!" % protein_2_embedding_filepath)
    prot_id_list = set()
    done_prot_id_list = set()
    with open(protein_2_embedding_filepath, "w") as wfp:
        writer = csv.writer(wfp)
        writer.writerow(["prot_id", "emb_filename"])
        for cur_protein_id_2_idx_filepath in protein_id_2_idx_filepath:
            with open(cur_protein_id_2_idx_filepath, "r") as rfp:
                reader = csv.reader(rfp)
                cnt = 0
                for row in reader:
                    cnt += 1
                    if cnt == 1:
                        continue
                    index, prot_id = row
                    prot_id_list.add(prot_id)
                    emb_path = os.path.join(embedding_dirpath, "%s.pt" % index)
                    if os.path.exists(emb_path):
                        writer.writerow([prot_id,  "%s.pt" % index])
                        done_prot_id_list.add(prot_id)
                    else:
                        raise Exception("emb_path :%s not exists" % emb_path)
                    if (cnt - 1) % 10000 == 0:
                        print("done %d" % (cnt - 1))
    print("embedding want to do: %d, done: %d, undo: %d" % (len(prot_id_list), len(done_prot_id_list), len(prot_id_list.difference(done_prot_id_list))))
    return protein_2_embedding_filepath


def merge(fasta_filepath, protein_2_pdb_filepath, protein_2_emb_filepath, label, source):
    structure = {}
    if protein_2_pdb_filepath and os.path.exists(protein_2_pdb_filepath):
        with open(protein_2_pdb_filepath, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1:
                    continue
                prot_id, pdb_filename, ptm, mean_plddt = row
                structure[prot_id] = [pdb_filename, ptm, mean_plddt]

    embedding = {}
    with open(protein_2_emb_filepath, "r") as rfp:
        reader = csv.reader(rfp)
        cnt = 0
        for row in reader:
            cnt += 1
            if cnt == 1:
                continue
            prot_id, emb_filename = row
            embedding[prot_id] = emb_filename

    prot_id_list, prot_id_2_seq = load_file(fasta_filepath)
    save_filepath = os.path.join(os.path.dirname(fasta_filepath), ".".join(os.path.basename(fasta_filepath).split(".")[:-1]) + "_with_pdb_emb.csv")
    print("save path: %s" % save_filepath)
    if os.path.exists(save_filepath):
        raise Exception("file: %s exists!" % save_filepath)
    with open(save_filepath, "w") as wfp:
        writer = csv.writer(wfp)
        writer.writerow(["prot_id", "seq", "seq_len", "pdb_filename", "ptm", "mean_plddt", "emb_filename", "label", "source"])
        stats = {"seq": 0, "pdb": 0, "emb": 0}
        for prot_id in prot_id_list:
            seq = prot_id_2_seq[prot_id].strip().strip("*")
            seq_len = len(seq)
            pdb_filename, ptm, mean_plddt = None, None, None
            emb_filename = None
            stats["seq"] += 1
            if prot_id in structure:
                pdb_filename, ptm, mean_plddt = structure[prot_id]
                stats["pdb"] += 1
            if prot_id in embedding:
                emb_filename = embedding[prot_id]
                stats["emb"] += 1
                writer.writerow([prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source])
        print("stats: ")
        print(stats)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fasta_filepath", default=None, required=True, type=str, help="fasta filepath")
parser.add_argument("--protein_id_2_pdb_idx_filepath", default=None, type=str, help="the protein id to pdb file index name filepath")
parser.add_argument("--pdb_dirpath", default=None, type=str, help="pdb file dirpath(every filename is index)")
parser.add_argument("--protein_id_2_emb_idx_filepath", default=None, required=True, type=str, help="the protein id to embedding file index name filepath")
parser.add_argument("--emb_dirpath", default=None, type=str, help="embedding file dirpath(every filename is index)")
parser.add_argument("--label", default=None, type=int, help="this dataset label")
parser.add_argument("--source", default=None, type=str, help="")
args = parser.parse_args()


if __name__ == "__main__":
    fasta_filepath = args.fasta_filepath
    protein_id_2_pdb_idx_filepath = args.protein_id_2_pdb_idx_filepath
    pdb_dirpath = args.pdb_dirpath
    protein_id_2_emb_idx_filepath = args.protein_id_2_emb_idx_filepath
    embedding_dirpath = args.embedding_dirpath
    protein_2_pdb_filepath = load_pdb(protein_id_2_pdb_idx_filepath, pdb_dirpath)
    protein_2_emb_filepath = load_emb(protein_id_2_emb_idx_filepath, embedding_dirpath)
    merge(fasta_filepath, protein_2_pdb_filepath, protein_2_emb_filepath, label=args.label, source=args.source)
    merge(fasta_filepath, protein_2_pdb_filepath, protein_2_emb_filepath, label=args.labe,  source=args.source)
