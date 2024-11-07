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
@datetime: 2022/12/10 20:18
@project: DeepProtFunc
@file: predict
@desc: predict batch data from file(the structural embedding information prepared in advance)
'''

import argparse, time
import os, sys, json, codecs
from subword_nmt.apply_bpe import BPE
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../src/common")
try:
    from common.metrics import *
    from common.multi_label_metrics import *
    from utils import set_seed, plot_bins, csv_reader
    from SSFN.model import *
    from data_loader import load_and_cache_examples, convert_examples_to_features, InputExample, InputFeatures
except ImportError:
    from src.common.metrics import *
    from src.common.multi_label_metrics import *
    from src.utils import set_seed, plot_bins, csv_reader
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
        seq_tokenizer = tokenizer_class.from_pretrained(os.path.join(model_dir, "sequence"),
                                                        do_lower_case=args.do_lower_case)
        # seq_tokenizer = tokenizer_class(os.path.join(model_dir, "sequence"), "vocab.txt"), do_lower_case=args.do_lower_case)
        if args.subword:
            bpe_codes_prot = codecs.open(args.codes_file)
            subword = BPE(bpe_codes_prot, merges=-1, separator='')
    else:
        seq_tokenizer = None

    if args.has_struct_encoder:
        struct_tokenizer = tokenizer_class.from_pretrained(os.path.join(model_dir, "struct"),
                                                           do_lower_case=args.do_lower_case)
        # struct_tokenizer = tokenizer_class(os.path.join(model_dir, "struct", "vocab.txt"), do_lower_case=args.do_lower_case)
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
        rows,
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
    :param rows: [[prot_id, protein_seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename], ...]
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
    for row in rows:
        # agreed 7 columns
        prot_id, protein_seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename = row[0:7]
        batch_info.append([prot_id, protein_seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename])
        if len(row) > 7:
            batch_info[-1].extend(row[7:])
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
            if os.path.exists(args.pdb_dir):
                pdb_filepath = os.path.join(args.pdb_dir, pdb_filename)
            else:
                pdb_filepath = os.path.join(args.data_path, "pdbs", pdb_filename)
            loaded = np.load(pdb_filepath, allow_pickle=True)
            cmap = loaded["C_alpha"] if args.cmap_type == "C_alpha" else loaded["C_beta"]
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
            if os.path.exists(args.emb_dir):
                embedding_filepath = os.path.join(args.emb_dir, emb_filename)
            else:
                embedding_filepath = os.path.join(args.data_path, "embs", emb_filename)
            try:
                embedding = torch.load(embedding_filepath)
            except Exception as e:
                print(prot_id, protein_seq)
                print("embedding_filepath:\n", embedding_filepath)
                print(e)
                raise Exception(e)
            if args.embedding_type == "contacts":
                embedding_info = embedding["contacts"].numpy()
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
                embedding_info = embedding["representations"][36].numpy()
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
                            embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (0, 0)], mode='constant',
                                                    constant_values=pad_token)
                        else:
                            embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length
                            embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, 0)], mode='constant',
                                                    constant_values=pad_token)
            elif args.embedding_type == "bos":
                embedding_info = embedding["bos_representations"][36].numpy()
                embedding_attention_mask = None
            else:
                raise Exception("Not support arg: --embedding_type=%s" % args.embedding_type)
        else:
            embedding_info = None
            embedding_attention_mask = None

        '''
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
        else:
            raise KeyError(args.task_type)
        '''
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
        batch_input["embedding_info"] = torch.tensor(np.array([f.embedding_info for f in features], dtype=np.float32),
                                                     dtype=torch.float32).to(args.device)
        if args.embedding_type != "bos":
            batch_input["embedding_attention_mask"] = torch.tensor([f.embedding_attention_mask for f in features],
                                                                   dtype=torch.long).to(args.device)

    return batch_info, batch_input


def predict_probs(
        args,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        rows
):
    '''
    prediction
    :param args:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param model:
    :param rows:
    :return:
    '''
    '''
    label_list = processor.get_labels(label_filepath=args.label_filepath)
    label_map = {label: i for i, label in enumerate(label_list)}
    '''
    batch_info, batch_input = transform_sample_2_feature(args, rows, seq_tokenizer, subword, struct_tokenizer)
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
        rows
):
    '''
    predict positive or negative label for a batch
    :param args:
    :param label_id_2_name:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param model:
    :param rows: n samples
    :return:
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, rows)
    # print("probs dim: ", probs.ndim)
    preds = (probs >= args.threshold).astype(int).flatten()
    res = []
    for idx, info in enumerate(batch_info):
        res.append([info[0], info[1], float(probs[idx][0]), label_id_2_name[preds[idx]], *info[2:]])
    return res


def predict_multi_class(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        rows
):
    '''
    predict a class for a batch
    :param args:
    :param label_id_2_name:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param model:
    :param rows: n samples
    :return:
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, rows)
    # print("probs dim: ", probs.ndim)
    preds = np.argmax(probs, axis=-1)
    res = []
    for idx, info in enumerate(batch_info):
        res.append([info[0], info[1], float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]], *info[2:]])
    return res


def predict_multi_label(
        args,
        label_id_2_name,
        seq_tokenizer,
        subword,
        struct_tokenizer,
        model,
        rows
):
    '''
    predict multi-labels for a batch
    :param args:
    :param label_id_2_name:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param model:
    :param rows: n samples
    :return:
    '''
    batch_info, probs = predict_probs(args, seq_tokenizer, subword, struct_tokenizer, model, rows)
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    res = []
    for idx, info in enumerate(batch_info):
        res.append(
            [
                info[0], info[1],
                [float(probs[idx][label_index]) for label_index in preds[idx]],
                [label_id_2_name[label_index] for label_index in preds[idx]], *info[2:]
            ]
        )
    return res


parser = argparse.ArgumentParser(description="Prediction for RdRP(embedding in advance)")
parser.add_argument("--torch_hub_dir", default=None, type=str,
                    help="set the torch hub dir path for saving pretrained model(default:~/.cache/torch/hub)")
parser.add_argument("--data_path", default=None, type=str, required=True,
                    help="the data filepath(if it is csv format, Column 0 must be id, Colunm 1 must be seq.")
parser.add_argument("--emb_dir", default=None, type=str,
                    help="the structural embedding file dir.")
parser.add_argument("--pdb_dir", default=None, type=str,
                    help="the 3d-structure pdb file dir.")
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
parser.add_argument("--evaluate", action="store_true",
                    help="whether to evaluate the predicted results.")
parser.add_argument("--ground_truth_col_index",  default=None, type=int,
                    help="the ground truth col inde of the ${data_path}, default: None.")
parser.add_argument("--threshold",  default=0.5, type=float,
                    help="sigmoid threshold for binary-class or multi-label classification, None for multi-class classification, default: 0.5.")
parser.add_argument("--batch_size",  default=16, type=int,
                    help="batch size per GPU/CPU for evaluatio, default: 16.")
parser.add_argument("--print_per_batch",  default=1000,
                    type=int,
                    help="how many batches are completed every time for printing progress information, default: 1000.")
parser.add_argument("--gpu_id", default=None, type=int,
                    help="the used gpu index, -1 for cpu")
args = parser.parse_args()

if args.torch_hub_dir is not None:
    if not os.path.exists(args.torch_hub_dir):
        os.makedirs(args.torch_hub_dir)
    os.environ['TORCH_HOME'] = args.torch_hub_dir

if __name__ == "__main__":
    model_dir = "../models/%s/%s/%s/%s/%s/%s" % (
        args.dataset_name, args.dataset_type, args.task_type,
        args.model_type, args.time_str,
        args.step if args.step == "best" else "checkpoint-{}".format(args.step)
    )
    config_dir = "../logs/%s/%s/%s/%s/%s" % (
        args.dataset_name, args.dataset_type, args.task_type,
        args.model_type,  args.time_str
    )
    predict_dir = "../predicts/%s/%s/%s/%s/%s/%s" % (
        args.dataset_name, args.dataset_type, args.task_type,
        args.model_type, args.time_str,
        args.step if args.step == "best" else "checkpoint-{}".format(args.step)
    )

    # Step1: loading the model configuration
    config = load_args(config_dir)
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
        # args.batch_size = config["per_gpu_eval_batch_size"]

        if args.task_type in ["multi-label", "multi_label"]:
            # to do
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

    col_names = ["protein_id", "seq", "predict_prob", "predict_label", "seq_len", "pdb_filename", "ptm", "mean_plddt", "emb_filename"]
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
        with open(args.data_path, "r") as rfp:
            for row in csv_reader(rfp, header=True, header_filter=False):
                if len(row) > 7:
                    col_names.extend(row[7:])
                    break
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
    else:
        raise Exception("not csv file.")
    # total batch number
    total_batch_num = (total_cnt + args.batch_size - 1 - had_done_cnt)//args.batch_size
    print("total num: %d, had done num: %d, batch size: %d, batch_num: %d" % (
        total_cnt,
        had_done_cnt,
        args.batch_size,
        total_batch_num
    ))
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
                res = predict_func(args, label_id_2_name, seq_tokenizer, subword, struct_tokenizer, model, row_batch)
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
                    print("done total: %d, p: %d, n: %d, per batch use time: %f" % (
                        done_batch_num * args.batch_size + had_done_cnt,
                        predict_stats["1"] if "1" in predict_stats else (predict_stats[1] if 1 in predict_stats else 0),
                        predict_stats["0"] if "0" in predict_stats else (predict_stats[0] if 0in predict_stats else 0),
                        use_time/done_batch_num
                    ))
        if len(row_batch) > 0:
            begin_time = time.time()
            res = predict_func(args, label_id_2_name, seq_tokenizer, subword, struct_tokenizer, model, row_batch)
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
            print("done total: %d, p: %d, n: %d, per batch use time: %f" % (
                total_cnt,
                predict_stats["1"] if "1" in predict_stats else (predict_stats[1] if 1 in predict_stats else 0),
                predict_stats["0"] if "0" in predict_stats else (predict_stats[0] if 0in predict_stats else 0),
                use_time/done_batch_num
            ))
        print("prediction done. total batch: %d, use time: %f." % (done_batch_num, use_time))

    # plot the Sequence Length Distribution
    seq_length_distribution_pic_savepath = os.path.join(predict_dir, "seq_length_distribution.png")
    if not os.path.exists(os.path.dirname(seq_length_distribution_pic_savepath)):
        os.makedirs(os.path.dirname(seq_length_distribution_pic_savepath))
    seq_len_list = []
    for item in seq_len_stats.items():
        seq_len_list.extend([item[0]] * item[1])
    plot_bins(
        seq_len_list,
        xlabel="sequence length",
        ylabel="distribution",
        bins=40,
        filepath=seq_length_distribution_pic_savepath
    )

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

        evaluate_metrics = evaluate_func(
            np.array(ground_truth_list),
            np.array(predict_pred_list),
            savepath=confusion_matrix_savepath
        )
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
            wfp.write("%s=%s\n" % (item[0], item[1]))
        wfp.write("#" * 50 + "\n")
        wfp.write("prediction statistics:\n")
        for item in sorted(predict_stats.items(), key=lambda x:x[0]):
            wfp.write("%s=%s\n" % (item[0], item[1]))
    print("-" * 25 + "predict stats:" + "-" * 25)
    print("ground truth: ")
    print(ground_truth_stats)
    print("prediction: ")
    print(predict_stats)



