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
@datetime: 2022/12/13 10:48
@project: DeepProtFunc
@file: embedding_from_esm_rdrp.py
@desc: predict the protein structural embedding from Meta-ESMFold
'''
import os, csv
import torch, sys
import argparse
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from esm import Alphabet, FastaBatchedDataset, pretrained, MSATransformer
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import write_fasta, fasta_reader, clean_seq
except ImportError:
    from src.utils import write_fasta, fasta_reader, clean_seq


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    # dim len: [640, 1280, 2560, 5120]
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t36_3B_UR50D",
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        choices=["esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]
    )
    parser.add_argument(
        "-name",
        type=str,
        default=None,
        help="sequence name.",
    )
    parser.add_argument(
        "-seq",
        type=str,
        default=None,
        help="sequence.",
    )
    parser.add_argument(
        '-i',
        "--file",
        type=str,
        help="FASTA/CSV file on which to extract representations",
    )
    parser.add_argument(
        '-o',
        "--output_dir",
        type=str,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=4094,
        help="truncate sequences longer than the given value",
    )
    parser.add_argument(
        "--try_failure",
        action="store_true",
        help="when CUDA Out of Memory, try to reduce the truncation_seq_length"
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def load_done_set(result_info_path, uncompleted_path, other_fasta_id_2_idx_file, other_uncompleted_file):
    """
    What has already been done does not need to be re-predict
    :param result_info_path:
    :param uncompleted_path:
    :param other_fasta_id_2_idx_file:
    :param other_uncompleted_file:
    :return:
    """
    done_set = set()
    max_uuid_index = 0
    if result_info_path and os.path.exists(result_info_path):
        with open(result_info_path, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1 or row[0] == "index":
                    continue
                index = int(row[0])
                uuid = row[1].strip()
                if max_uuid_index < index:
                    max_uuid_index = index
                done_set.add(uuid)
    if other_fasta_id_2_idx_file and os.path.exists(other_fasta_id_2_idx_file):
        with open(other_fasta_id_2_idx_file, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1 or row[0] == "index":
                    continue
                index = int(row[0])
                uuid = row[1].strip()
                if max_uuid_index < index:
                    max_uuid_index = index
                done_set.add(uuid)
    if uncompleted_path and os.path.exists(uncompleted_path):
        with open(uncompleted_path, "r") as rfp:
            for line in rfp:
                line = line.strip()
                ridx = line.rfind(",")
                if ridx > -1:
                    uuid = line[:ridx]
                    done_set.add(uuid)
    if other_uncompleted_file and os.path.exists(other_uncompleted_file):
        with open(other_uncompleted_file, "r") as rfp:
            for line in rfp:
                line = line.strip()
                ridx = line.rfind(",")
                if ridx > -1:
                    uuid = line[:ridx]
                    done_set.add(uuid)
    return done_set, max_uuid_index


def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_name)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        # print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.file)
    '''
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    '''
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length))

    print(f"Read {args.file} with {len(dataset)} sequences")
    os.makedirs(args.output_dir,  exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    had = False
    if os.path.exists(args.uncompleted_file):
        uncompleted_wfp = open(args.uncompleted_file, "a+")
    else:
        uncompleted_wfp = open(args.uncompleted_file, "w")
    if os.path.exists(args.fasta_id_2_idx_file):
        fasta_id_2_idx_wfp = open(args.fasta_id_2_idx_file, "a+")
        had = True
    else:
        fasta_id_2_idx_wfp = open(args.fasta_id_2_idx_file, "w")
    fasta_id_2_idx_writer = csv.writer(fasta_id_2_idx_wfp)
    if not had:
        fasta_id_2_idx_writer.writerow(["index", "uuid"])
    protein_idx = args.begin_uuid_index
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            # strs ori sequence，toks: processed sequence（such as truncating，padding）
            protein_ids, strs, toks = batch
            protein_ids = [">" + v.strip() if v and v[0] != ">" else v.strip() for v in protein_ids]
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            try:
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            except RuntimeError as e:
                if e.args[0].startswith("CUDA out of memory"):
                    if len(strs) > 1:
                        print(
                            f"Failed (CUDA out of memory) to predict batch of size {len(strs)}. "
                            "Try lowering `--toks_per_batch."
                        )
                    else:
                        print(
                            f"Failed (CUDA out of memory) on sequence {protein_ids[0]} of length {len(strs[0])}."
                        )
                    for idx, v in enumerate(protein_ids):
                        uncompleted_wfp.write("%s,%d\n" % (v, len(strs[idx])))
                        uncompleted_wfp.flush()
                continue

            # logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for idx, protein_id in enumerate(protein_ids):
                protein_idx += 1
                cur_output_file = os.path.join(args.output_dir,  "%s.pt" % protein_idx)
                result = {"protein_id": protein_id, "seq": strs[idx], "seq_len": len(strs[idx]), "max_len": args.truncation_seq_length}

                truncate_len = min(args.truncation_seq_length, len(strs[idx]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[idx, 1: truncate_len + 1].clone() for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[idx, 1: truncate_len + 1].mean(0).clone() for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[idx, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[idx, 1: truncate_len + 1, 1: truncate_len + 1].clone()

                torch.save(
                    result,
                    cur_output_file,
                )
                fasta_id_2_idx_writer.writerow([protein_idx, protein_id])
                fasta_id_2_idx_wfp.flush()
    uncompleted_wfp.close()
    fasta_id_2_idx_wfp.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    total_example = 0
    proteins = {}
    args.output_dir = os.path.join(args.output_dir, str(args.model_name))
    input_type = "file"
    if not args.file and args.seq:
        input_type = "not file"
        args.file = "self_input.fasta"
    elif not args.file and not args.seq:
        raise Exception("-file or -seq")
    protein_idx = 0
    point_idx = args.file.rfind(".")
    if point_idx > 0:
        args.fasta_id_2_idx_file = args.file[0: point_idx] + "_embed_fasta_id_2_idx.csv"
        if args.try_failure:
            args.uncompleted_file = args.file[0: point_idx] + "_embed_uncompleted_%d.txt" % args.truncation_seq_length
        else:
            args.uncompleted_file = args.file[0: point_idx] + "_embed_uncompleted.txt"
    else:
        args.fasta_id_2_idx_file = args.file + "_embed_fasta_id_2_idx.csv"
        if args.try_failure:
            args.uncompleted_file = args.file + "_embed_uncompleted_%d.txt" % args.truncation_seq_length
        else:
            args.uncompleted_file = args.file + "_embed_uncompleted.txt"
    args.other_fasta_id_2_idx_file = os.path.join(os.path.dirname(os.path.dirname(args.file)), ("_".join(os.path.basename(args.file).split("_")[:-1]) if "_" in os.path.basename(args.file) else ".".join(os.path.basename(args.file).split(".")[:-1])) + "_embed_fasta_id_2_idx.csv")
    args.other_uncompleted_file = os.path.join(os.path.dirname(os.path.dirname(args.file)), ("_".join(os.path.basename(args.file).split("_")[:-1]) if "_" in os.path.basename(args.file) else ".".join(os.path.basename(args.file).split(".")[:-1])) + "_embed_uncompleted.txt")
    print("other_fasta_id_2_idx_file: %s" % args.other_fasta_id_2_idx_file)
    print("other_uncompleted_file: %s" % args.other_uncompleted_file)
    # try failure samples
    if args.try_failure:
        done_set, begin_uuid_index = load_done_set(args.fasta_id_2_idx_file, None, args.other_fasta_id_2_idx_file, None)
    else:
        done_set, begin_uuid_index = load_done_set(args.fasta_id_2_idx_file, args.uncompleted_file, args.other_fasta_id_2_idx_file, args.other_uncompleted_file)
    print("done_set: %d" % len(done_set))
    print("begin_uuid_index: %d" % begin_uuid_index)
    # the begin index
    args.begin_uuid_index = begin_uuid_index
    # the fasta need to calc
    new_fasta_file = args.file.replace(".csv", "").replace(".fasta", "") + "_need_embed_%d.fasta" % (args.begin_uuid_index + 1)
    if os.path.exists(new_fasta_file):
        os.remove(new_fasta_file)
        # raise Exception(new_fasta_file + " exists.")
    sequence_list = []
    protein_num = {}
    if input_type == "file":
        if args.file.endswith(".csv"):
            with open(args.file, "r") as rfp:
                reader = csv.reader(rfp)
                cnt = 0
                for row in reader:
                    cnt += 1
                    if cnt == 1:
                        continue
                    protein_id, seq = row[0].strip(), row[1].strip()
                    seq = clean_seq(protein_id, seq)
                    if protein_id and protein_id[0] != ">":
                        protein_id = ">" + protein_id
                    # remove duplication
                    if protein_id in proteins:
                        if proteins[protein_id] != seq:
                            protein_num[protein_id] += 1
                            protein_id = protein_id + "_append_%d" % protein_num[protein_id]
                        else:
                            continue
                    else:
                        protein_num[protein_id] = 1
                        proteins[protein_id] = seq
                    if protein_id in done_set:
                        continue
                    sequence_list.append(SeqRecord(Seq(seq, None), id=protein_id[1:] if protein_id and protein_id[0] == ">" else protein_id, description=""))
                    total_example += 1
        else:
            for obj in fasta_reader(args.file):
                protein_id = obj[0].strip()
                seq = obj[1].strip()
                seq = clean_seq(protein_id, seq)
                # remove duplication
                if protein_id in proteins:
                    if proteins[protein_id] != seq:
                        protein_num[protein_id] += 1
                        protein_id = protein_id + "_append_%d" % protein_num[protein_id]
                    else:
                        continue
                else:
                    protein_num[protein_id] = 1
                    proteins[protein_id] = seq
                if protein_id in done_set:
                    continue
                sequence_list.append(SeqRecord(Seq(seq, None), id=protein_id[1:] if protein_id and protein_id[0] == ">" else protein_id, description=""))
                total_example += 1
    else:
        seqs = args.seq.split(",")
        total_example = len(seqs)
        names = args.name.split(",")
        assert len(seqs) == len(names)
        sequence_list = []
        for protein_id, seq in zip(names, seqs):
            if protein_id in done_set:
                continue
            sequence_list.append(SeqRecord(Seq(seq, None), id=protein_id[1:] if protein_id and protein_id[0] == ">" else protein_id, description=""))
    write_fasta(new_fasta_file, sequence_list)
    args.file = new_fasta_file
    args.total_example = total_example
    assert len(sequence_list) == args.total_example
    print("Total examples: %d" % args.total_example)

    main(args)

    # recommend the length list: 4094 2046 1982 1790 1534 1278 1150 1022
