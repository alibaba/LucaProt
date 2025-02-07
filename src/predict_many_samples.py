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
@datetime: 2023/4/10 18:26
@project: DeepProtFunc
@file: predict_many_samples
@desc: predict many samples from file
'''
import argparse
import csv
import numpy as np
import os, sys, json, codecs
from subword_nmt.apply_bpe import BPE
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from common.multi_label_metrics import *
    from protein_structure.predict_structure import predict_embedding, predict_pdb, calc_distance_maps
    from utils import set_seed, plot_bins, csv_reade, fasta_reader, clean_seq
    from SSFN.model import *
    from data_loader import load_and_cache_examples, convert_examples_to_features, InputExample, InputFeatures
except ImportError:
    from src.common.multi_label_metrics import *
    from src.protein_structure.predict_structure import predict_embedding, predict_pdb, calc_distance_maps
    from src.utils import set_seed, plot_bins, csv_reader, fasta_reader, clean_seq
    from src.SSFN.model import *
    from src.data_loader import load_and_cache_examples, convert_examples_to_features, InputExample, InputFeatures

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
    filename = "../dataset/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type, filename)
    if filename and os.path.exists(filename):
        with open(filename, "r") as rfp:
            for line in rfp:
                strs = line.strip().split("###")
                label_code_2_name[strs[0]] = strs[1]
    return label_code_2_name


def load_args(log_dir):
    '''
    load model running args
    :param log_dir:
    :return: config
    '''
    print("-" * 25 + "log dir: " + "-" * 25)
    print(log_dir)
    print("-" * 60)
    log_filepath = os.path.join(log_dir, "logs.txt")
    if not os.path.exists(log_filepath):
        raise Exception("%s not exists" % log_filepath)
    with open(log_filepath, "r") as rfp:
        for line in rfp:
            if line.startswith("{"):
                obj = json.loads(line.strip())
                return obj
    return {}


def load_model(args, model_dir):
    '''
    load the model
    :param args:
    :param model_dir:
    :return:
    '''
    # load tokenizer and model
    device = torch.device(args.device)
    config_class, model_class, tokenizer_class = BertConfig, SequenceAndStructureFusionNetwork, BertTokenizer

    # config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r"), encoding="UTF-8"))
    config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r")))
    # for sequence
    subword = None
    if args.has_seq_encoder:
        seq_tokenizer = tokenizer_class.from_pretrained(
            os.path.join(model_dir, "sequence"),
            do_lower_case=args.do_lower_case
        )
        if args.subword:
            bpe_codes_prot = codecs.open(args.codes_file)
            subword = BPE(bpe_codes_prot, merges=-1, separator='')
    else:
        seq_tokenizer = None

    if args.has_struct_encoder:
        struct_tokenizer = tokenizer_class.from_pretrained(
            os.path.join(model_dir, "struct"),
            do_lower_case=args.do_lower_case
        )
    else:
        struct_tokenizer = None

    model = model_class.from_pretrained(model_dir, args=args)

    model.to(device)
    model.eval()

    # load labels
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

    print("-" * 25 + "label_id_2_name: " + "-" * 25)
    if len(label_id_2_name) < 20:
        print(label_id_2_name)
    print("label size: ", len(label_id_2_name))
    print("-" * 60)

    return config, subword, seq_tokenizer, struct_tokenizer, model, label_id_2_name, label_name_2_id


def transform_sample_2_feature(
        args,
        row,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True
):
    '''
    batch sample transform to batch input
    :param args:
    :param row: [protein_id, seq]
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param pad_on_left:
    :param pad_token:
    :param pad_token_segment_id:
    :param mask_padding_with_zero:
    :return:
    '''
    features = []
    batch_info = []
    # id, seq
    prot_id, protein_seq = row[0], row[1]
    batch_info.append(row)
    assert seq_tokenizer is not None or struct_tokenizer is not None or args.embedding_type is not None
    if seq_tokenizer:
        if subword:
            seq_to_list = subword.process_line(protein_seq).split(" ")
        else:
            seq_to_list = [v for v in protein_seq]
        cur_seq_len = len(seq_to_list)
        if cur_seq_len > args.seq_max_length - 2:
            if args.trunc_type == "left":
                seq_to_list = seq_to_list[2 - args.seq_max_length:]
            else:
                seq_to_list = seq_to_list[:args.seq_max_length - 2]
        seq = " ".join(seq_to_list)
        inputs = seq_tokenizer.encode_plus(
            seq,
            None,
            add_special_tokens=True,
            max_length=args.seq_max_length,
            truncation=True
        )
        # input_ids: token index list
        # token_type_ids: token type index list
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        real_token_len = len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.seq_max_length - len(input_ids)
        attention_mask_padding_length = padding_length

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * attention_mask_padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * attention_mask_padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == args.seq_max_length, "Error with input length {} vs {}".format(len(input_ids), args.seq_max_length)
        assert len(attention_mask) == args.seq_max_length, "Error with input length {} vs {}".format(len(attention_mask), args.seq_max_length)
        assert len(token_type_ids) == args.seq_max_length, "Error with input length {} vs {}".format(len(token_type_ids), args.seq_max_length)
    else:
        input_ids = None
        attention_mask = None
        token_type_ids = None
        real_token_len = None
    if struct_tokenizer:
        # for structure
        cur_seq_len = len(protein_seq)
        seq_list = [ch for ch in protein_seq]
        if cur_seq_len > args.struct_max_length:
            if args.trunc_type == "left":
                seq_list = seq_list[-args.struct_max_length:]
            else:
                seq_list = seq_list[:args.struct_max_length]
        seq = " ".join(seq_list)
        inputs = struct_tokenizer.encode_plus(
            seq,
            None,
            add_special_tokens=False,
            max_length=args.struct_max_length,
            truncation=True,
            return_token_type_ids=False,
        )
        struct_input_ids = inputs["input_ids"]
        real_struct_node_size = len(struct_input_ids)
        padding_length = args.struct_max_length - real_struct_node_size if real_struct_node_size < args.struct_max_length else 0
        pdb, mean_plddt, ptm, processed_seq = predict_pdb([prot_id, protein_seq], args.trunc_type, num_recycles=4, truncation_seq_length=args.truncation_seq_length, chunk_size=64, cpu_type="cpu-offload")
        # if the savepath not exists, create it
        if args.pdb_dir:
            if not os.path.exists(args.pdb_dir):
                os.makedirs(args.pdb_dir)
            pdb_filepath = os.path.join(args.pdb_dir, prot_id.replace("/", "_") + ".pdb")
            with open(pdb_filepath, "w") as wfp:
                wfp.write(pdb)
        c_alpha, c_beta = calc_distance_maps(pdb, args.chain, processed_seq)
        cmap = c_alpha[args.chain]['contact-map'] if args.cmap_type == "C_alpha" else c_beta[args.chain]['contact-map']
        # use the specific threshold to transform the float contact map into 0-1 contact map
        cmap = np.less_equal(cmap, args.cmap_thresh).astype(np.int32)
        struct_contact_map = cmap
        real_shape = struct_contact_map.shape
        if real_shape[0] > args.struct_max_length:
            if args.trunc_type == "left":
                struct_contact_map = struct_contact_map[-args.struct_max_length:, -args.struct_max_length:]
            else:
                struct_contact_map = struct_contact_map[:args.struct_max_length, :args.struct_max_length]
            contact_map_padding_length = 0
        else:
            contact_map_padding_length = args.struct_max_length - real_shape[0]
        assert contact_map_padding_length == padding_length

        if contact_map_padding_length > 0:
            if pad_on_left:
                struct_input_ids = [pad_token] * padding_length + struct_input_ids
                struct_contact_map = np.pad(struct_contact_map, [(contact_map_padding_length, 0), (contact_map_padding_length, 0)], mode='constant', constant_values=pad_token)
            else:
                struct_input_ids = struct_input_ids + ([pad_token] * padding_length)
                struct_contact_map = np.pad(struct_contact_map, [(0, contact_map_padding_length), (0, contact_map_padding_length)], mode='constant', constant_values=pad_token)

        assert len(struct_input_ids) == args.struct_max_length, "Error with input length {} vs {}".format(len(struct_input_ids), args.struct_max_length)
        assert struct_contact_map.shape[0] == args.struct_max_length, "Error with input length {}x{} vs {}x{}".format(struct_contact_map.shape[0], struct_contact_map.shape[1], args.struct_max_length, args.struct_max_length)
    else:
        struct_input_ids = None
        struct_contact_map = None
        real_struct_node_size = None

    if args.embedding_type:
        # for embedding
        embedding_info, processed_seq = predict_embedding(
            [prot_id, protein_seq],
            args.trunc_type,
            "representations" if args.embedding_type == "matrix" else args.embedding_type,
            repr_layers=[-1],
            truncation_seq_length=args.truncation_seq_length - 2,
            device=args.device
        )
        # failure on GPU, then using CPU for embedding
        if embedding_info is None:
            # 失败,则调用cpu进行embedding推理
            embedding_info, processed_seq = predict_embedding(
                [prot_id, protein_seq],
                args.trunc_type,
                "representations" if args.embedding_type == "matrix" else args.embedding_type,
                repr_layers=[-1],
                truncation_seq_length=args.truncation_seq_length - 2,
                device=torch.device("cpu")
            )
        if args.emb_dir:
            if not os.path.exists(args.emb_dir):
                os.makedirs(args.emb_dir)

            embedding_filepath = os.path.join(args.emb_dir, prot_id.replace("/", "_") + ".pt")
            torch.save(embedding_info, embedding_filepath)
        if args.embedding_type == "contacts":
            emb_l = embedding_info.shape[0]
            embedding_attention_mask = [1 if mask_padding_with_zero else 0] * emb_l
            if emb_l > args.embedding_max_length:
                if args.trunc_type == "left":
                    embedding_info = embedding_info[-args.embedding_max_length:, -args.embedding_max_length:]
                else:
                    embedding_info = embedding_info[:args.embedding_max_length, :args.embedding_max_length]
                embedding_attention_mask = [1 if mask_padding_with_zero else 0] * args.embedding_max_length
            else:
                embedding_padding_length = args.embedding_max_length - emb_l
                if embedding_padding_length > 0:
                    if pad_on_left:
                        embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask
                        embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (embedding_padding_length, 0)], mode='constant', constant_values=pad_token)
                    else:
                        embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length
                        embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, embedding_padding_length)], mode='constant', constant_values=pad_token)
        elif args.embedding_type == "matrix":
            emb_l = embedding_info.shape[0]
            embedding_attention_mask = [1 if mask_padding_with_zero else 0] * emb_l
            if emb_l > args.embedding_max_length:
                if args.trunc_type == "left":
                    embedding_info = embedding_info[-args.embedding_max_length:, :]
                else:
                    embedding_info = embedding_info[:args.embedding_max_length, :]
                embedding_attention_mask = [1 if mask_padding_with_zero else 0] * args.embedding_max_length
            else:
                embedding_padding_length = args.embedding_max_length - emb_l
                if embedding_padding_length > 0:
                    if pad_on_left:
                        embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask
                        embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (0, 0)], mode='constant', constant_values=pad_token)
                    else:
                        embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length
                        embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, 0)], mode='constant', constant_values=pad_token)
        elif args.embedding_type == "bos":
            embedding_attention_mask = None
        else:
            raise Exception("Not support arg: --embedding_type=%s" % args.embedding_type)
    else:
        embedding_info = None
        embedding_attention_mask = None
    features.append(
        InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            real_token_len=real_token_len,
            struct_input_ids=struct_input_ids,
            struct_contact_map=struct_contact_map,
            real_struct_node_size=real_struct_node_size,
            embedding_info=embedding_info,
            embedding_attention_mask=embedding_attention_mask,
            label=None
        )
    )
    batch_input = {}
    # "labels": torch.tensor([f.label for f in features], dtype=torch.long).to(args.device),
    if seq_tokenizer:
        batch_input.update(
            {
                "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device),
                "attention_mask": torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(args.device),
                "token_type_ids": torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(args.device),
            }
        )
    if struct_tokenizer:
        batch_input.update(
            {
                "struct_input_ids": torch.tensor([f.struct_input_ids for f in features], dtype=torch.long).to(args.device),
                "struct_contact_map": torch.tensor([f.struct_contact_map for f in features], dtype=torch.long).to(args.device),
            }
        )
    if args.embedding_type:
        batch_input["embedding_info"] = torch.tensor(np.array([f.embedding_info for f in features], dtype=np.float32), dtype=torch.float32).to(args.device)
        if args.embedding_type != "bos":
            batch_input["embedding_attention_mask"] = torch.tensor([f.embedding_attention_mask for f in features], dtype=torch.long).to(args.device)

    return batch_info, batch_input


def predict_probs(
        args,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):
    '''
    prediction for one sample
    :param args:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer
    :param model:
    :param row: one sample
    :return:
    '''
    '''
    label_list = processor.get_labels(label_filepath=args.label_filepath)
    label_map = {label: i for i, label in enumerate(label_list)}
    '''
    # in order to be able to embed longer sequences
    model.to(torch.device("cpu"))
    batch_info, batch_input = transform_sample_2_feature(args, row, seq_tokenizer, subword, struct_tokenizer)
    model.to(args.device)
    if torch.cuda.is_available():
        probs = model(**batch_input)[1].detach().cpu().numpy()
    else:
        probs = model(**batch_input)[1].detach().numpy()
    return batch_info, probs


def predict_binary_class(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):
    '''
    predict positive or negative label for one sample
    :param args:
    :param label_id_2_name:
    :param seq_tokenizer
    :param subword:
    :param struct_tokenizer
    :param model:
    :param row: one sample
    :return:
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, row)
    # print("probs dim: ", probs.ndim)
    preds = (probs >= args.threshold).astype(int).flatten()
    res = []
    for idx, info in enumerate(batch_info):
        cur_res = [info[0], info[1], float(probs[idx][0]), label_id_2_name[preds[idx]]]
        if len(info) > 2:
            cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_multi_class(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):
    '''
    predict multi-labels for one sample
    :param args:
    :param label_id_2_name:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param model:
    :param row: one sample
    :return:
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, row)
    # print("probs dim: ", probs.ndim)
    preds = np.argmax(probs, axis=-1)
    res = []
    for idx, info in enumerate(batch_info):
        cur_res = [info[0], info[1], float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]
        if len(info) > 2:
            cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_multi_label(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        row
):
    '''
    predict multi-labels for one sample
    :param args:
    :param label_id_2_name:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param model:
    :param row: one sample
    :return:
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, row)
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    res = []
    for idx, info in enumerate(batch_info):
        cur_res = [
            info[0],
            info[1],
            [float(probs[idx][label_index]) for label_index in preds[idx]],
            [label_id_2_name[label_index] for label_index in preds[idx]]
        ]
        if len(info) > 2:
            cur_res += info[2:]
        res.append(cur_res)
    return res


def main():
    parser = argparse.ArgumentParser(description="Prediction RdRP")
    parser.add_argument("--torch_hub_dir", default=None, type=str,
                        help="set the torch hub dir path for saving pretrained model(default:~/.cache/torch/hub)")
    parser.add_argument("--fasta_file", default=None, type=str, required=True,
                        help="fasta file path")
    parser.add_argument("--save_file", default=None, type=str, required=True,
                        help="the result file path")
    parser.add_argument("--truncation_seq_length", default=4096, type=int, required=True,
                        help="truncation seq length(include: [CLS] and [SEP]")
    parser.add_argument("--emb_dir", default=None, type=str,
                        help="the llm embedding save dir. default: None")
    parser.add_argument("--pdb_dir", default="protein", type=str,
                        help="the 3d-structure pdb save dir. default: None")
    parser.add_argument("--chain", default=None, type=str,
                        help="pdb chain for contact map computing")
    parser.add_argument("--dataset_name", default="rdrp_40_extend", type=str, required=True,
                        help="the dataset name for model building.")
    parser.add_argument("--dataset_type", default="protein", type=str, required=True,
                        help="the dataset type for model building.")
    parser.add_argument("--task_type", default=None, type=str, required=True,
                        choices=["multi_label", "multi_class", "binary_class"],
                        help="the task type for model building.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="model type.")
    parser.add_argument("--time_str", default=None, type=str, required=True,
                        help="the running time string(yyyymmddHimiss) of model building.")
    parser.add_argument("--step", default=None, type=str, required=True,
                        help="the training global step of model finalization.")
    parser.add_argument("--threshold",  default=0.5, type=float,
                        help="sigmoid threshold for binary-class or multi-label classification, None for multi-class classification, defualt: 0.5.")
    parser.add_argument("--print_per_number", default=100, type=int,
                        help="print per number")
    parser.add_argument("--gpu_id", default=None, type=int, help="the used gpu index, -1 for cpu")
    input_args = parser.parse_args()
    return input_args


if __name__ == "__main__":
    args = main()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if args.torch_hub_dir is not None:
        if not os.path.exists(args.torch_hub_dir):
            os.makedirs(args.torch_hub_dir)
        os.environ['TORCH_HOME'] = args.torch_hub_dir
    if not os.path.exists(args.fasta_file):
        print("the input fasta file: %s not exists!" % args.fasta_file)
    if os.path.exists(args.save_file):
        print("the output file: %s exists!" % args.save_file)
    else:
        dirpath = os.path.dirname(args.save_file)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    model_dir = "%s/../models/%s/%s/%s/%s/%s/%s" % (
        SCRIPT_DIR, args.dataset_name, args.dataset_type, args.task_type,
        args.model_type, args.time_str,
        args.step if args.step == "best" else "checkpoint-{}".format(args.step)
    )
    config_dir = "%s/../logs/%s/%s/%s/%s/%s" % (
        SCRIPT_DIR, args.dataset_name, args.dataset_type, args.task_type,
        args.model_type,  args.time_str
    )
    predict_dir = "%s/../predicts/%s/%s/%s/%s/%s/%s" % (
        SCRIPT_DIR, args.dataset_name, args.dataset_type, args.task_type,
        args.model_type, args.time_str,
        args.step if args.step == "best" else "checkpoint-{}".format(args.step)
    )

    # Step1: loading the model configuration
    config = load_args(config_dir)
    for key, value in config.items():
        try:
            if value.startswith("../"):
                value = os.path.join(SCRIPT_DIR, value)
        except AttributeError:
            continue
        print(f'My item {value} is labelled {key}')
        config[key] = value
    print("-" * 25 + "config:" + "-" * 25)
    print(config)
    print("-" * 60)
    if config:
        args.dataset_name = config["dataset_name"]
        args.dataset_type = config["dataset_type"]
        args.task_type = config["task_type"]
        args.model_type = config["model_type"]
        args.has_seq_encoder = config["has_seq_encoder"]
        args.has_struct_encoder = config["has_struct_encoder"]
        args.has_embedding_encoder = config["has_embedding_encoder"]
        args.subword = config["subword"]
        args.codes_file = config["codes_file"]
        args.input_mode = config["input_mode"]
        args.label_filepath = config["label_filepath"]
        if not os.path.exists(args.label_filepath):
            args.label_filepath = os.path.join(config_dir, "label.txt")
        args.output_dir = config["output_dir"]
        args.config_path = config["config_path"]

        args.do_lower_case = config["do_lower_case"]
        args.sigmoid = config["sigmoid"]
        args.loss_type = config["loss_type"]
        args.output_mode = config["output_mode"]

        args.seq_vocab_path = config["seq_vocab_path"]
        args.seq_pooling_type = config["seq_pooling_type"]
        args.seq_max_length = config["seq_max_length"]
        args.struct_vocab_path = config["struct_vocab_path"]
        args.struct_max_length = config["struct_max_length"]
        args.struct_pooling_type = config["struct_pooling_type"]
        args.trunc_type = config["trunc_type"]
        args.no_position_embeddings = config["no_position_embeddings"]
        args.no_token_type_embeddings = config["no_token_type_embeddings"]
        args.cmap_type = config["cmap_type"]
        args.cmap_type = float(config["cmap_thresh"])
        args.embedding_input_size = config["embedding_input_size"]
        args.embedding_pooling_type = config["embedding_pooling_type"]
        args.embedding_max_length = config["embedding_max_length"]
        args.embedding_type = config["embedding_type"]
        if args.task_type in ["multi-label", "multi_label"]:
            args.sigmoid = True
        elif args.task_type in ["binary-class", "binary_class"]:
            args.sigmoid = True

    if args.gpu_id == -1:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("-" * 25 + "args:" + "-" * 25)
    print(args.__dict__.items())
    print("-" * 60)
    print("-" * 25 + "model_dir list:" + "-" * 25)
    print(os.listdir(model_dir))
    print("-" * 60)

    if args.device.type == 'cpu':
        print("Running Device is CPU!")
    else:
        print("Running Device is GPU!")
    print("-" * 60)

    # Step2: loading the tokenizer and model
    config, subword, seq_tokenizer, struct_tokenizer, model, label_id_2_name, label_name_2_id = \
        load_model(args=args, model_dir=model_dir)
    predict_func = None
    if args.task_type in ["multi-label", "multi_label"]:
        predict_func = predict_multi_label
    elif args.task_type in ["binary-class", "binary_class"]:
        predict_func = predict_binary_class
    elif args.task_type in ["multi-class", "multi_class"]:
        predict_func = predict_multi_class
    else:
        raise Exception("Not Support Task Type: %s" % args.task_type)
    done = 0
    with open(args.save_file, "w") as wfp:
        writer = csv.writer(wfp)
        writer.writerow(["protein_id", "seq", "prob", "label"])
        for row in fasta_reader(args.fasta_file):
            # Step 3: prediction
            row = [row[0], clean_seq(row[0], row[1])]
            res = predict_func(args, label_id_2_name, seq_tokenizer, subword, struct_tokenizer, model, row)
            writer.writerow(res[0])
            done += 1
            if done % args.print_per_number == 0:
                print("done : %d" % done)
    print("all done: %d" % done)



