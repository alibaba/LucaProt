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
@datetime: 2022/11/26 20:34
@project: DeepProtFunc
@file: data_loader
@desc: loading dataset from file or tfrecords(large dataset)
'''
import math
import torch
import numpy as np
import os, copy, json, csv, sys
from torch.utils.data import TensorDataset
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset
try:
    from utils import plot_bins, load_labels, file_reader
except ImportError:
    from src.utils import plot_bins, load_labels, file_reader
import logging
logger = logging.getLogger(__name__)


class DataProcessor(object):
    '''
    Data Processor
    '''
    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file, header=False,  filter_header=True):
        if input_file.endswith(".csv"):
            return cls._read_csv(input_file, header=header, filter_header=filter_header)
        else:
            return cls._read_txt(input_file, header=header, filter_header=filter_header)

    @classmethod
    def _read_csv(cls, input_file, header=True, filter_header=True):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            # reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            delimiter = ","
            if input_file.endswith(".tsv"):
                delimiter = "\t"
            reader = csv.reader(f, delimiter=delimiter)
            lines = []
            for line in reader:
                # Check if the version of python is 2
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            if header and filter_header:
                lines = lines[1:]
            return lines

    @classmethod
    def _read_txt(cls, input_file, header=False, filter_header=True):
        with open(input_file, "r", encoding="utf-8-sig") as rfp:
            lines = []
            for line in rfp:
                lines.append(line.strip().split("\t"))
            if header and filter_header:
                lines = lines[1:]
            return lines


class InputExample(object):
    '''
    One Example Building，the inputs include: guid，seq，contact map matrix, structural embedding info, embedding_len, embedding_dim, label
    '''
    def __init__(self, guid, seq, contact_map=None, embedding_info=None, embedding_len=None, embedding_dim=None, label=None):
        self.guid = guid
        self.seq = seq
        self.contact_map = contact_map
        self.embedding_info = embedding_info
        self.embedding_len = embedding_len
        self.embedding_dim = embedding_dim
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    '''
    Create the features of one example, the inputs include: token ids，seq attention mask，seq type ids,structure node ids，structure contact map, structure node size, structural embedding_info, structural embedding attention mask, label, global attention mask
    '''
    def __init__(self, input_ids, attention_mask, token_type_ids, real_token_len,
                 struct_input_ids, struct_contact_map, real_struct_node_size,
                 embedding_info, embedding_attention_mask,
                 label, global_attention_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.real_token_len = real_token_len
        self.struct_input_ids = struct_input_ids
        self.struct_contact_map = struct_contact_map
        self.real_struct_node_size = real_struct_node_size
        self.embedding_info = embedding_info
        self.embedding_attention_mask = embedding_attention_mask
        self.label = label
        self.global_attention_mask = global_attention_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SequenceStructureProcessor(DataProcessor):
    '''
    Sequence and Structure Processor(
    The storage format of the dataset
    separate_file=True, indicates that the sample id is separate from the information of the sample, otherwise, all fields are in a csv file, and there is a header
    train.txt no header, each row is an id
    dev.txt no header, each row is an id
    test.txt no header, each row is an id
    label.txt label list file, exists header
    npz/${id}.npz: dict, the sample corresponding to the id, which includes multiple information fields, get the info by label_type

    '''
    def __init__(self, model_type, separate_file, filename_pattern=None):
        '''
        :param model_type:
        :param separate_file:
        :param filename_pattern:
        '''
        self.separate_file = separate_file
        self.filename_pattern = filename_pattern
        self.model_type = model_type

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            uuid=tensor_dict['idx'].numpy(),
            seq=tensor_dict['seq'].numpy(),
            contact_map=tensor_dict['contact_map'].numpy(),
            embedding_info=tensor_dict['embedding_info'].numpy(),
            label=str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, cmap_type, cmap_thresh, embedding_type):
        if self.filename_pattern is not None and os.path.exists(os.path.join(data_dir, self.filename_pattern.format("train"))):
            filepath = os.path.join(data_dir, self.filename_pattern.format("train"))
            if filepath.endswith(".txt"):
                header = False
            elif filepath.endswith(".csv"):
                header = True
        else:
            if os.path.exists(os.path.join(data_dir, "train.txt")):
                filepath = os.path.join(data_dir, "train.txt")
                header = False
            elif os.path.exists(os.path.join(data_dir, "train.csv")):
                filepath = os.path.join(data_dir, "train.csv")
                header = True
            else:
                raise Exception("not exists train.* in %s" % data_dir)
        datasets = self._create_examples(self._read_file(filepath, header=header, filter_header=True), "train", data_dir,
                                         cmap_type, cmap_thresh, embedding_type, self.separate_file)
        # shuffle
        '''
        for _ in range(5):
            random.shuffle(datasets)
        '''
        return datasets

    def get_dev_examples(self, data_dir, cmap_type, cmap_thresh, embedding_type):
        if self.filename_pattern is not None and os.path.exists(os.path.join(data_dir, self.filename_pattern.format("dev"))):
            filepath = os.path.join(data_dir, self.filename_pattern.format("dev"))
            if filepath.endswith(".txt"):
                header = False
            elif filepath.endswith(".csv"):
                header = True
        else:
            if os.path.exists(os.path.join(data_dir, "dev.txt")):
                filepath = os.path.join(data_dir, "dev.txt")
                header = False
            elif os.path.exists(os.path.join(data_dir, "dev.csv")):
                filepath = os.path.join(data_dir, "dev.csv")
                header = True
            else:
                raise Exception("not exists dev.* in %s" % data_dir)
        return self._create_examples(self._read_file(filepath, header=header, filter_header=True), "dev", data_dir,
                                     cmap_type, cmap_thresh, embedding_type, self.separate_file)

    def get_test_examples(self, data_dir, cmap_type, cmap_thresh, embedding_type):
        if self.filename_pattern is not None and os.path.exists(os.path.join(data_dir, self.filename_pattern.format("test"))):
            filepath = os.path.join(data_dir, self.filename_pattern.format("test"))
            if filepath.endswith(".txt"):
                header = False
            elif filepath.endswith(".csv"):
                header = True
        else:
            if os.path.exists(os.path.join(data_dir, "test.txt")):
                filepath = os.path.join(data_dir, "test.txt")
                header = False
            elif os.path.exists(os.path.join(data_dir, "test.csv")):
                filepath = os.path.join(data_dir, "test.csv")
                header = True
            else:
                raise Exception("not exists test.* in %s" % data_dir)
        return self._create_examples(self._read_file(filepath, header=header, filter_header=True), "test", data_dir,
                                     cmap_type, cmap_thresh, embedding_type, self.separate_file)

    def get_labels(self, label_filepath):
        '''
        get labels from file, exists header
        :param label_filepath:
        :return:
        '''
        with open(label_filepath, "r") as fp:
            labels = []
            multi_cols = False
            cnt = 0
            for line in fp.readlines():
                cnt += 1
                if cnt == 1:
                    if line.find(",") > 0:
                        multi_cols = True
                    continue
                line = line.strip()
                if multi_cols:
                    idx = line.find(",")
                    if idx > 0:
                        label_name = line[idx+1:].strip()
                    else:
                        label_name = line
                else:
                    label_name = line
                labels.append(label_name)
            return labels

    @staticmethod
    def _create_examples_bak(lines, dataset_type, data_dir, cmap_type, cmap_thresh, embedding_type, separate_file):
        examples = []
        if separate_file:
            for row in lines:
                # prot_id, seq, pdb_index, embed_idx, label = row[0], row[1], row[2], row[3], row[4]
                prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
                # pdb_filepath = os.path.join(data_dir, "npz", "%s.npz" % pdb_index)

                cmap = None
                if cmap_type:
                    pdb_filepath = os.path.join(data_dir, "pdbs", pdb_filename)
                    if pdb_filename and os.path.isfile(pdb_filepath):
                        loaded = np.load(pdb_filepath, allow_pickle=True)
                        prot_id = loaded["prot_id"].item()
                        cmap = loaded["C_alpha"] if cmap_type == "C_alpha" else loaded["C_beta"]
                        # convect the real distance matrix into 0-1 contact map
                        cmap = np.less_equal(cmap, cmap_thresh).astype(np.int32)
                        # seqres = loaded["seqres"].item()
                embedding_info = None
                if embedding_type:
                    # embedding_filepath = os.path.join(data_dir, "embeds", "%s.pt" % embed_idx)
                    embedding_filepath = os.path.join(data_dir, "embs", emb_filename)
                    if emb_filename and os.path.isfile(embedding_filepath):
                        emb = torch.load(embedding_filepath)
                        embedding_len = emb["seq_len"]
                        if embedding_type == "contacts":
                            embedding_info = emb["contacts"].numpy()
                            embedding_len = embedding_info.shape[0]
                            embedding_d = embedding_info.shape[1]
                        elif embedding_type == "matrix":
                            embedding_info = emb["representations"][36].numpy()
                            embedding_len = embedding_info.shape[0]
                            embedding_d = embedding_info.shape[1]
                        elif embedding_type == "bos":
                            embedding_info = emb["bos_representations"][36].numpy()
                            embedding_d = embedding_info.shape[0]
                    else:
                        raise Exception("%s not exists." % embedding_filepath)
                examples.append(InputExample(guid=dataset_type + "#" + prot_id,
                                             seq=seq,
                                             contact_map=cmap,
                                             embedding_info=embedding_info,
                                             embedding_len=embedding_len,
                                             embedding_dim=embedding_d,
                                             label=label))
        else:
            for row in lines:
                prot_id = row[0]
                seqres = row[1]
                label = row[2]
                examples.append(InputExample(guid=dataset_type + "#" + prot_id, seq=seqres, contact_map=None, embedding_info=None, label=label))
        return examples

    @staticmethod
    def _create_examples(lines, dataset_type, data_dir, cmap_type, cmap_thresh, embedding_type, separate_file):
        examples = []
        if separate_file:
            for row in lines:
                # prot_id, seq, pdb_index, embed_idx, label = row[0], row[1], row[2], row[3], row[4]
                prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
                # pdb_filepath = os.path.join(data_dir, "npz", "%s.npz" % pdb_index)

                cmap = None
                if cmap_type:
                    pdb_filepath = os.path.join(data_dir, "pdbs", pdb_filename)
                    if pdb_filename and os.path.isfile(pdb_filepath):
                        loaded = np.load(pdb_filepath, allow_pickle=True)
                        prot_id = loaded["prot_id"].item()
                        cmap = loaded["C_alpha"] if cmap_type == "C_alpha" else loaded["C_beta"]
                        # convect real distance matrix into 0-1 contact map
                        cmap = np.less_equal(cmap, cmap_thresh).astype(np.int32)
                        # seqres = loaded["seqres"].item()
                embedding_info = None
                embedding_len = -1
                embedding_d = -1
                if embedding_type:
                    # embedding_filepath = os.path.join(data_dir, "embeds", "%s.pt" % embed_idx)
                    embedding_filepath = os.path.join(data_dir, "embs", emb_filename)
                    if emb_filename and os.path.isfile(embedding_filepath):
                        emb = torch.load(embedding_filepath)
                        embedding_len = emb["seq_len"]
                        if embedding_type == "contacts":
                            embedding_info = emb["contacts"].numpy()
                            embedding_len = embedding_info.shape[0]
                            embedding_d = embedding_info.shape[1]
                        elif embedding_type == "matrix":
                            embedding_info = emb["representations"][36].numpy()
                            embedding_len = embedding_info.shape[0]
                            embedding_d = embedding_info.shape[1]
                        elif embedding_type == "bos":
                            embedding_info = emb["bos_representations"][36].numpy()
                            embedding_d = embedding_info.shape[0]
                    else:
                        raise Exception("%s not exists." % embedding_filepath)
                '''
                examples.append(InputExample(guid=dataset_type + "#" + prot_id,
                                             seq=seq,
                                             contact_map=cmap,
                                             embedding_info=embedding_info,
                                             embedding_len=embedding_len,
                                             embedding_dim=embedding_d,
                                             label=label))
                '''
                yield InputExample(guid=dataset_type + "#" + prot_id,
                                   seq=seq,
                                   contact_map=cmap,
                                   embedding_info=embedding_info,
                                   embedding_len=embedding_len,
                                   embedding_dim=embedding_d,
                                   label=label)
        else:
            for row in lines:
                prot_id = row[0]
                seqres = row[1]
                label = row[2]
                # examples.append(InputExample(guid=dataset_type + "#" + prot_id, seq=seqres, contact_map=None, embedding_info=None, label=label))
                yield InputExample(guid=dataset_type + "#" + prot_id, seq=seqres, contact_map=None, embedding_info=None, label=label)
        # return examples


def convert_examples_to_features(examples,
                                 subword,
                                 seq_tokenizer,
                                 struct_tokenizer,
                                 seq_max_length=512,
                                 struct_max_length=512,
                                 embedding_type=None,
                                 embedding_max_length=512,
                                 output_mode=None,
                                 label_list=None,
                                 label_filepath=None,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 trunc_type="right"):
    '''
    a sample to features
    :param examples:
    :param subword:
    :param tokenizer:
    :param struct_tokenizer:
    :param seq_max_length:
    :param struct_max_length:
    :param output_mode:
    :param label_list:
    :param label_filepath:
    :param pad_on_left:
    :param pad_token:
    :param pad_token_segment_id:
    :param mask_padding_with_zero:
    :param trunc_type:
    :return:
    '''
    if label_list is None:
        label_list = load_labels(filepath=label_filepath, header=True)

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    seq_len_list = []

    # iterate over all samples
    ex_index = 0
    for example in examples:
        # for (ex_index, example) in enumerate(examples):
        ex_index += 1
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        seq_max_length = int(seq_max_length)
        if seq_tokenizer:
            # for sequence
            seq = example.seq
            seq_len = len(seq)
            seq_len_list.append(seq_len)
            # tokenization
            if subword:
                seq_to_list = subword.process_line(seq).split(" ")
            else:
                seq_to_list = [v for v in seq]
            cur_seq_len = len(seq_to_list)
            if cur_seq_len > seq_max_length - 2:
                if trunc_type == "left":
                    seq_to_list = seq_to_list[2 - seq_max_length:]
                else:
                    seq_to_list = seq_to_list[:seq_max_length - 2]
            seq = " ".join(seq_to_list)
            inputs = seq_tokenizer.encode_plus(
                seq,
                None,
                add_special_tokens=True,
                max_length=seq_max_length,
                truncation=True
            )
            # input_ids
            # token_type_ids
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            real_token_len = len(input_ids)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = seq_max_length - len(input_ids)
            attention_mask_padding_length = padding_length

            if pad_on_left:
                input_ids = [pad_token] * padding_length + input_ids
                attention_mask = [0 if mask_padding_with_zero else 1] * attention_mask_padding_length + attention_mask
                token_type_ids = [pad_token_segment_id] * padding_length + token_type_ids
            else:
                input_ids = input_ids + [pad_token] * padding_length
                attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * attention_mask_padding_length
                token_type_ids = token_type_ids + [pad_token_segment_id] * padding_length

            assert len(input_ids) == seq_max_length, "Error with input length {} vs {}".format(len(input_ids), seq_max_length)
            assert len(attention_mask) == seq_max_length, "Error with input length {} vs {}".format(len(attention_mask), seq_max_length)
            assert len(token_type_ids) == seq_max_length, "Error with input length {} vs {}".format(len(token_type_ids), seq_max_length)
        else:
            input_ids = None
            attention_mask = None
            token_type_ids = None
            real_token_len = None

        if struct_tokenizer is None:
            struct_input_ids = None
            struct_contact_map = None
            real_struct_node_size = None
        else:
            # for structure
            cur_seq_len = len(example.seq)
            seq_list = [ch for ch in example.seq]
            if cur_seq_len > struct_max_length:
                if trunc_type == "left":
                    seq_list = seq_list[-struct_max_length:]
                else:
                    seq_list = seq_list[:struct_max_length]
            seq = " ".join(seq_list)
            inputs = struct_tokenizer.encode_plus(
                seq,
                None,
                add_special_tokens=False,
                max_length=struct_max_length,
                truncation=True,
                return_token_type_ids=False,
            )
            struct_input_ids = inputs["input_ids"]
            real_struct_node_size = len(struct_input_ids)
            padding_length = struct_max_length - real_struct_node_size if real_struct_node_size < struct_max_length else 0

            struct_contact_map = example.contact_map
            real_shape = struct_contact_map.shape
            if real_shape[0] > struct_max_length:
                if trunc_type == "left":
                    struct_contact_map = struct_contact_map[-struct_max_length:, -struct_max_length:]
                else:
                    struct_contact_map = struct_contact_map[:struct_max_length, :struct_max_length]
                contact_map_padding_length = 0
            else:
                contact_map_padding_length = struct_max_length - real_shape[0]
            assert contact_map_padding_length == padding_length

            if contact_map_padding_length > 0:
                if pad_on_left:
                    struct_input_ids = [pad_token] * padding_length + struct_input_ids
                    struct_contact_map = np.pad(struct_contact_map, [(contact_map_padding_length, 0), (contact_map_padding_length, 0)], mode='constant', constant_values=pad_token)
                else:
                    struct_input_ids = struct_input_ids + ([pad_token] * padding_length)
                    struct_contact_map = np.pad(struct_contact_map, [(0, contact_map_padding_length), (0, contact_map_padding_length)], mode='constant', constant_values=pad_token)

            assert len(struct_input_ids) == struct_max_length, "Error with input length {} vs {}".format(len(struct_input_ids), struct_max_length)
            assert struct_contact_map.shape[0] == struct_max_length, "Error with input length {}x{} vs {}x{}".format(struct_contact_map.shape[0], struct_contact_map.shape[1], struct_max_length, struct_max_length)
        if embedding_type:
            # for embedding
            emb_l = example.embedding_len
            emb_d = example.embedding_dim
            embedding_attention_mask = [1 if mask_padding_with_zero else 0] * emb_l
            if embedding_type == "contacts":
                embedding_info = example.embedding_info
                if emb_l > embedding_max_length:
                    if trunc_type == "left":
                        embedding_info = embedding_info[-embedding_max_length:, -embedding_max_length:]
                    else:
                        embedding_info = embedding_info[:embedding_max_length, :embedding_max_length]
                    embedding_attention_mask = [1 if mask_padding_with_zero else 0] * embedding_max_length
                else:
                    embedding_padding_length = embedding_max_length - emb_l
                    if embedding_padding_length > 0:
                        if pad_on_left:
                            embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask
                            embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (embedding_padding_length, 0)], mode='constant', constant_values=pad_token)
                        else:
                            embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length
                            embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, embedding_padding_length)], mode='constant', constant_values=pad_token)

            elif embedding_type == "matrix":
                embedding_info = example.embedding_info
                if emb_l > embedding_max_length:
                    if trunc_type == "left":
                        embedding_info = embedding_info[-embedding_max_length:, :]
                    else:
                        embedding_info = embedding_info[:embedding_max_length, :]
                    embedding_attention_mask = [1 if mask_padding_with_zero else 0] * embedding_max_length
                else:
                    embedding_padding_length = embedding_max_length - emb_l
                    if embedding_padding_length > 0:
                        if pad_on_left:
                            embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask
                            embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (0, 0)], mode='constant', constant_values=pad_token)
                        else:
                            embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length
                            embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, 0)], mode='constant', constant_values=pad_token)
            elif embedding_type == "bos":
                embedding_info = example.embedding_info
                embedding_attention_mask = None
        else:
            embedding_info = None
            embedding_attention_mask = None

        # for label
        if output_mode in ["multi-class", "multi_class"]:
            label_name = label_map[example.label]
            if isinstance(label_name, str):
                label = label_map[label_name]
            else:
                label = label_name
        elif output_mode == "regression":
            label = float(example.label)
        elif output_mode in ["multi-label", "multi_label"]:
            if isinstance(example.label, str):
                label = [0] * len(label_map)
                for label_name in eval(example.label):
                    if isinstance(label_name, str):
                        label_id = label_map[label_name]
                    else:
                        label_id = label_name
                    label[label_id] = 1
            else:
                label = [0] * len(label_map)
                for label_name in example.label:
                    if isinstance(label_name, str):
                        label_id = label_map[label_name]
                    else:
                        label_id = label_name
                    label[label_id] = 1
        elif output_mode in ["binary-class", "binary_class"]:
            label_name = label_map[example.label]
            if isinstance(label_name, str):
                label = label_map[label_name]
            else:
                label = label_name
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("real_sequence: %s" % example.seq)
            logger.info("real_sequence_len: %s" % len(example.seq))
            if subword:
                logger.info("subword_tokens: %s" % seq)
                logger.info("subword_tokens_len: %s" % real_token_len)
            if seq_tokenizer:
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if struct_tokenizer:
                logger.info("struct_input_ids: %s" % " ".join([str(x) for x in struct_input_ids]))
                logger.info("struct_node_real_size: %s" % real_struct_node_size)
                logger.info("struct_node_size: %s" % len(struct_input_ids))
                logger.info("struct_contact_map: %s" % str(struct_contact_map))
                logger.info("struct_contact_map_real_shape: %s" % str(real_shape))
                logger.info("struct_contact_map_shape: %s" % str(struct_contact_map.shape))
            if embedding_type:
                logger.info("embedding_info: %s" % " ".join([str(x) for x in embedding_info]))
                if embedding_attention_mask:
                    logger.info("embedding_attention_mask: %s" % " ".join([str(x) for x in embedding_attention_mask]))
            if output_mode in ["multi-label", "multi_label"]:
                logger.info("label: %s (id = %s)" % (str(example.label), str(label)))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label))
        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          real_token_len=real_token_len,
                          struct_input_ids=struct_input_ids,
                          struct_contact_map=struct_contact_map,
                          real_struct_node_size=real_struct_node_size,
                          embedding_info=embedding_info,
                          embedding_attention_mask=embedding_attention_mask,
                          label=label
                          )
        )
    return features, seq_len_list


def load_and_cache_examples(args, processor, seq_tokenizer, subword, struct_tokenizer, evaluate=False, predict=False, log_fp=None):
    """
    transform the dataset into features，and cache them into cached_features_file
    :param args.cmap_type: [C_alpha, C_beta]
    :param args.label_type: onts = ['molecular_function', 'biological_process', 'cellular_component', 'rdrp']
    """
    if processor is None:
        processor = SequenceStructureProcessor(model_type=args.model_type, separate_file=args.separate_file, filename_pattern=args.filename_pattern)
    output_mode = args.task_type
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
    # cached_features_file: cached directory for the constructed features of the dataset
    if evaluate:
        exec_model = "dev"
    elif predict:
        exec_model = "test"
    else:
        exec_model = "train"

    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_{}_{}_{}_{}_{}{}_vocab{}{}{}{}{}{}{}".format(
        exec_model,
        str(args.seq_max_length),
        str(args.trunc_type),
        args.dataset_name,
        args.dataset_type,
        args.task_type,
        args.model_type,
        args.input_mode,
        args.label_type,
        "_subword" if args.subword else "",
        str(seq_tokenizer.vocab_size) if seq_tokenizer else "",
        str(args.struct_max_length) if struct_tokenizer else "",
        "_vocab" + str(struct_tokenizer.vocab_size) if struct_tokenizer else "",
        "_" + args.label_type if struct_tokenizer else "",
        "_" + args.cmap_type if struct_tokenizer else "",
        "_" + args.embedding_type if args.embedding_type else "",
        "_" + str(args.embedding_max_length) if args.embedding_type else "",
    ))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s\n", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features into cached file %s\n", cached_features_file)
        label_list = processor.get_labels(label_filepath=args.label_filepath)
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, args.cmap_type, args.cmap_thresh, args.embedding_type)
            seq_len_distribution_savepath = "../pics/%s/%s/%s/%s/%s_seq_len_distribution.png" %(args.dataset_name, args.dataset_type, args.task_type, args.model_type, "dev")
            '''
            if log_fp:
                log_fp.write("dev examples num: %d\n" % len(examples))
                log_fp.write("#" * 100 + "\n")
            print("dev examples num: %d" % len(examples))
            '''
        elif predict:
            examples = processor.get_test_examples(args.data_dir, args.cmap_type, args.cmap_thresh, args.embedding_type)
            seq_len_distribution_savepath = "../pics/%s/%s/%s/%s/%s_seq_len_distribution.png" %(args.dataset_name, args.dataset_type, args.task_type, args.model_type, "test")
            '''
            if log_fp:
                log_fp.write("dev examples num: %d\n" % len(examples))
                log_fp.write("#" * 100 + "\n")
            print("test examples num: %d" % len(examples))
            '''
        else:
            examples = processor.get_train_examples(args.data_dir, args.cmap_type, args.cmap_thresh, args.embedding_type)
            seq_len_distribution_savepath = "../pics/%s/%s/%s/%s/%s_seq_len_distribution.png" %(args.dataset_name, args.dataset_type, args.task_type, args.model_type, "train")
            '''
            if log_fp:
                log_fp.write("train examples num: %d\n" % len(examples))
                log_fp.write("#" * 100 + "\n")
            print("train examples num: %d" % len(examples))
            '''
        if seq_tokenizer:
            logger.info("tokenizer vocab size: %d\n", seq_tokenizer.vocab_size)
        features, seq_len_list = convert_examples_to_features(examples,
                                                              subword=subword,
                                                              seq_tokenizer=seq_tokenizer,
                                                              struct_tokenizer=struct_tokenizer,
                                                              seq_max_length=args.seq_max_length,
                                                              struct_max_length=args.struct_max_length,
                                                              embedding_type=args.embedding_type,
                                                              embedding_max_length=args.embedding_max_length,
                                                              output_mode=args.output_mode,
                                                              label_list=label_list,
                                                              label_filepath=args.label_filepath,
                                                              pad_on_left=False,
                                                              pad_token=seq_tokenizer.convert_tokens_to_ids([seq_tokenizer.pad_token])[0] if seq_tokenizer else 0,
                                                              pad_token_segment_id=0,
                                                              mask_padding_with_zero=True,
                                                              trunc_type=args.trunc_type
                                                              )
        del examples
        if not os.path.exists(os.path.dirname(seq_len_distribution_savepath)):
            os.makedirs(os.path.dirname(seq_len_distribution_savepath))
        plot_bins(seq_len_list, xlabel="sequence length", ylabel="distribution", bins=40, filepath=seq_len_distribution_savepath)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    res = []
    if seq_tokenizer:
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_real_token_len = torch.tensor([f.real_token_len for f in features], dtype=torch.long)
        res += [all_input_ids, all_attention_mask, all_token_type_ids, all_real_token_len]
    if struct_tokenizer:
        all_struct_input_ids = torch.tensor([f.struct_input_ids for f in features], dtype=torch.long)
        all_struct_contact_map = torch.tensor(np.array([f.struct_contact_map for f in features], dtype=np.int), dtype=torch.long)
        all_real_struct_node_size = torch.tensor([f.real_struct_node_size for f in features], dtype=torch.long)
        res += [all_struct_input_ids, all_struct_contact_map, all_real_struct_node_size]
    if args.embedding_type:
        all_embedding_info = torch.tensor(np.array([f.embedding_info for f in features], dtype=np.float32), dtype=torch.float32)
        if args.embedding_type != "bos":
            all_embedding_attention_mask = torch.tensor([f.embedding_attention_mask for f in features], dtype=torch.long)
            res += [all_embedding_info, all_embedding_attention_mask]
        else:
            res += [all_embedding_info]

    if output_mode in ["multi_class", "multi-class"]:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    elif output_mode in ["multi-label", "multi_label"]:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    res += [all_labels]
    dataset = TensorDataset(*res)
    return dataset


def convert_one_example_to_features(example,
                                    subword,
                                    seq_tokenizer,
                                    struct_tokenizer,
                                    seq_max_length=1024,
                                    struct_max_length=1024,
                                    embedding_type=None,
                                    embedding_max_length=1024,
                                    output_mode=None,
                                    label_map=None,
                                    label_list=None,
                                    label_filepath=None,
                                    pad_on_left=False,
                                    pad_token=0,
                                    pad_token_segment_id=0,
                                    mask_padding_with_zero=True,
                                    trunc_type="right"):
    '''
    convect one sample to features
    :param example:
    :param subword:
    :param seq_tokenizer:
    :param struct_tokenizer:
    :param seq_max_length:
    :param struct_max_length:
    :param output_mode:
    :param embedding_type:
    :param label_map:
    :param label_list:
    :param label_filepath:
    :param pad_on_left:
    :param pad_token:
    :param pad_token_segment_id:
    :param mask_padding_with_zero:
    :param trunc_type:
    :return:
    '''
    if label_map is None:
        if label_list is None:
            label_list = load_labels(filepath=label_filepath, header=True)
        label_map = {label: i for i, label in enumerate(label_list)}
    seq_max_length = int(seq_max_length)
    seq = example["seq"]
    seq_len = len(seq)
    if seq_tokenizer is None:
        input_ids = None
        attention_mask = None
        token_type_ids = None
        real_token_len = None
    else:
        # for sequence
        # tokenization
        if subword:
            seq_to_list = subword.process_line(seq).split(" ")
        else:
            seq_to_list = [v for v in seq]
        cur_seq_len = len(seq_to_list)
        if cur_seq_len > seq_max_length - 2:
            if trunc_type == "left":
                seq_to_list = seq_to_list[2 - seq_max_length:]
            else:
                seq_to_list = seq_to_list[:seq_max_length - 2]
        seq = " ".join(seq_to_list)
        inputs = seq_tokenizer.encode_plus(
            seq,
            None,
            add_special_tokens=True,
            max_length=seq_max_length,
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
        padding_length = seq_max_length - len(input_ids)
        attention_mask_padding_length = padding_length

        if pad_on_left:
            input_ids = [pad_token] * padding_length + input_ids
            attention_mask = [0 if mask_padding_with_zero else 1] * attention_mask_padding_length + attention_mask
            token_type_ids = [pad_token_segment_id] * padding_length + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * attention_mask_padding_length
            token_type_ids = token_type_ids + [pad_token_segment_id] * padding_length

        assert len(input_ids) == seq_max_length, "Error with input length {} vs {}".format(len(input_ids), seq_max_length)
        assert len(attention_mask) == seq_max_length, "Error with input length {} vs {}".format(len(attention_mask), seq_max_length)
        assert len(token_type_ids) == seq_max_length, "Error with input length {} vs {}".format(len(token_type_ids), seq_max_length)

    if struct_tokenizer is None:
        struct_input_ids = None
        struct_contact_map = None
        struct_contact_map_len = None
    else:
        # for structure
        seq_list = [ch for ch in example["seq"]]
        cur_seq_len = len(example["seq"])
        if cur_seq_len > struct_max_length:
            if trunc_type == "left":
                seq_list = seq_list[-struct_max_length:]
            else:
                seq_list = seq_list[:struct_max_length]
        seq = " ".join(seq_list)
        inputs = struct_tokenizer.encode_plus(
            seq,
            None,
            add_special_tokens=False,
            max_length=struct_max_length,
            truncation=True,
            return_token_type_ids=False,
        )
        struct_input_ids = inputs["input_ids"]
        struct_contact_map_len = len(struct_input_ids)
        padding_length = struct_max_length - struct_contact_map_len if struct_contact_map_len < struct_max_length else 0

        struct_contact_map = example["contact_map"]
        real_shape = example["contact_map_shape"]
        real_shape = (real_shape, real_shape)
        if real_shape[0] > struct_max_length:
            if trunc_type == "left":
                struct_contact_map = struct_contact_map[-struct_max_length:, -struct_max_length:]
            else:
                struct_contact_map = struct_contact_map[:struct_max_length, :struct_max_length]
            contact_map_padding_length = 0
        else:
            contact_map_padding_length = struct_max_length - real_shape[0]
        assert contact_map_padding_length == padding_length
        if contact_map_padding_length > 0:
            if pad_on_left:
                struct_input_ids = [pad_token] * padding_length + struct_input_ids
                struct_contact_map = np.pad(struct_contact_map, [(contact_map_padding_length, 0), (contact_map_padding_length, 0)], mode='constant', constant_values=pad_token)
            else:
                struct_input_ids = struct_input_ids + [pad_token] * padding_length
                struct_contact_map = np.pad(struct_contact_map, [(0, contact_map_padding_length), (0, contact_map_padding_length)], mode='constant', constant_values=pad_token)

        assert len(struct_input_ids) == struct_max_length, "Error with input length {} vs {}".format(len(struct_input_ids), struct_max_length)
        assert struct_contact_map.shape[0] == struct_max_length, "Error with input length {}x{} vs {}x{}".format(struct_contact_map.shape[0], struct_contact_map.shape[1], struct_max_length, struct_max_length)
    if embedding_type:
        emb_l = example["emb_l"]
        emb_d = example["emb_d"]
        embedding_attention_mask = [1 if mask_padding_with_zero else 0] * emb_l
        if embedding_type == "contacts" and example["contacts"]:
            embedding_info = example["contacts"]
            if emb_l > embedding_max_length:
                if trunc_type == "left":
                    embedding_info = embedding_info[-embedding_max_length:, -embedding_max_length:]
                else:
                    embedding_info = embedding_info[:embedding_max_length, :embedding_max_length]
                embedding_attention_mask = [1 if mask_padding_with_zero else 0] * embedding_max_length
            else:
                embedding_padding_length = embedding_max_length - emb_l
                if embedding_padding_length > 0:
                    if pad_on_left:
                        embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask
                        embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (embedding_padding_length, 0)], mode='constant', constant_values=pad_token)
                    else:
                        embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length
                        embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, embedding_padding_length)], mode='constant', constant_values=pad_token)
        elif embedding_type == "matrix":
            embedding_info = example["representations"]
            if emb_l > embedding_max_length:
                if trunc_type == "left":
                    embedding_info = embedding_info[-embedding_max_length:, :]
                else:
                    embedding_info = embedding_info[:embedding_max_length, :]
                embedding_attention_mask = [1 if mask_padding_with_zero else 0] * embedding_max_length
            else:
                embedding_padding_length = embedding_max_length - emb_l
                if embedding_padding_length > 0:
                    if pad_on_left:
                        embedding_attention_mask = [0 if mask_padding_with_zero else 1] * embedding_padding_length + embedding_attention_mask
                        embedding_info = np.pad(embedding_info, [(embedding_padding_length, 0), (0, 0)], mode='constant', constant_values=pad_token)
                    else:
                        embedding_attention_mask = embedding_attention_mask + [0 if mask_padding_with_zero else 1] * embedding_padding_length
                        embedding_info = np.pad(embedding_info, [(0, embedding_padding_length), (0, 0)], mode='constant', constant_values=pad_token)
        elif embedding_type == "bos":
            embedding_info = example["bos_representations"]
            embedding_attention_mask = None

    # for label
    if output_mode in ["multi-class", "multi_class"]:
        label_name = example["label"]
        if isinstance(label_name, str):
            label = label_map[label_name]
        else:
            label = label_name
    elif output_mode == "regression":
        label = float(example["label"])
    elif output_mode in ["multi-label", "multi_label"]:
        if isinstance(example["label"], str):
            label = [0] * len(label_map)
            for label_name in eval(example["label"]):
                if isinstance(label_name, str):
                    label_id = label_map[label_name]
                else:
                    label_id = label_name
                label[label_id] = 1
        else:
            label = [0] * len(label_map)
            for label_name in example["label"]:
                if isinstance(label_name, str):
                    label_id = label_map[label_name]
                else:
                    label_id = label_name
                label[label_id] = 1
    elif output_mode in ["binary-class", "binary_class"]:
        label_name = example["label"]
        if isinstance(label_name, str):
            label = label_map[label_name]
        else:
            label = label_name
    else:
        raise KeyError(output_mode)
    res = []
    if seq_tokenizer:
        res += [torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long), torch.tensor([real_token_len], dtype=torch.long)]
    if struct_tokenizer:
        res += [torch.tensor(struct_input_ids, dtype=torch.long), torch.tensor(struct_contact_map, dtype=torch.long),
                torch.tensor([struct_contact_map_len], dtype=torch.long)]
    if embedding_type:
        if embedding_type != "bos":
            res += [torch.tensor(embedding_info, dtype=torch.float32), torch.tensor(embedding_attention_mask, dtype=torch.long)]
        else:
            res += [torch.tensor(embedding_info, dtype=torch.float32)]
    res += [torch.tensor(label, dtype=torch.long)]
    return tuple(res)


def parse_tfrecord(single_record, subword, seq_tokenizer, struct_tokenizer, seq_max_length, struct_max_length,
                   embedding_type, embedding_max_length,
                   output_mode, label_map, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero, trunc_type, cmap_type, cmap_thresh):
    '''
    parse a tf-record
    :param single_record:
    :param subword:
    :param seq_tokenizer:
    :param struct_tokenizer:
    :param seq_max_length:
    :param struct_max_length:
    :param embedding_type:
    :param embedding_max_length:
    :param output_mode:
    :param label_map:
    :param pad_on_left:
    :param pad_token:
    :param pad_token_segment_id:
    :param mask_padding_with_zero:
    :param trunc_type:
    :param cmap_type:
    :param cmap_thresh:
    :return:
    '''
    '''
    id
    seq
    L
    bos_representations
    representations
    contacts
    C_alpha_dist_matrix
    C_beta_dist_matrix
    label
    '''
    id = "".join([chr(v) for v in single_record["id"]])
    # seq
    seq = "".join([chr(v) for v in single_record["seq"]])

    # embedding
    contacts = None
    bos_representations = None
    representations = None
    if embedding_type:
        embedding_len = single_record["emb_l"][0]
        '''
        print("seq len: %d" % len(seq))
        print("embedding_len: %d" % embedding_len)
        '''
        embedding_d = single_record["emb_size"][0]
        if "contacts" in single_record:
            contacts = single_record["contacts"]
            leng = math.sqrt(len(contacts))
            embedding_len = int(np.sqrt(leng))
            contacts = np.reshape(contacts, (embedding_len, embedding_len)).astype(np.float32)
        else:
            contacts = None
        bos_representations = single_record["bos_representations"]
        bos_representations = np.reshape(bos_representations, embedding_d).astype(np.float32)
        representations = single_record["representations"]
        representations = np.reshape(representations, (-1, embedding_d)).astype(np.float32)
        embedding_len = representations.shape[0]

    # pdb
    contact_map = None
    contact_map_shape = None
    if cmap_type is not None:
        contact_map = single_record["%s_dist_matrix" % cmap_type]
        contact_map_shape = single_record["pdb_l"][0]
        contact_map = np.reshape(contact_map, (contact_map_shape, contact_map_shape)).astype(np.float32)
        contact_map = np.less_equal(contact_map, cmap_thresh).astype(np.int32)

    label = single_record["label"]

    example = {'id': id,
               'seq': seq,
               'contacts': contacts,
               "bos_representations": bos_representations,
               "representations": representations,
               "emb_l": embedding_len,
               "emb_d": embedding_d,
               'contact_map': contact_map,
               "contact_map_shape": contact_map_shape,
               "label": label
               }
    record = convert_one_example_to_features(
        example,
        subword,
        seq_tokenizer,
        struct_tokenizer,
        seq_max_length,
        struct_max_length,
        embedding_type,
        embedding_max_length,
        output_mode=output_mode,
        label_map=label_map,
        label_list=None,
        label_filepath=None,
        pad_on_left=pad_on_left,
        pad_token=pad_token,
        pad_token_segment_id=pad_token_segment_id,
        mask_padding_with_zero=mask_padding_with_zero,
        trunc_type=trunc_type
    )
    return record


def load_and_cache_examples_for_tfrecords(args, processor:DataProcessor, seq_tokenizer, subword, struct_tokenizer, evaluate=False, predict=False, log_fp=None):
    '''
    load tfrecords
    :param args:
    :param processor:
    :param seq_tokenizer:
    :param subword:
    :param struct_tokenizer:
    :param evaluate:
    :param predict:
    :param log_fp:
    :return:
    '''
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
    # cached_features_file: cached directory for the constructed features of the dataset
    suffix = ("_pdb_emb" if args.has_struct_encoder and args.has_embedding_encoder else ("_pdb" if args.has_struct_encoder else "_emb" if args.has_embedding_encoder else ""))
    if args.multi_tfrecords:
        tfrecords_name = "{}%s.tfrecords" % suffix
        index_pattern = "{}%s.index" % suffix
    else:
        tfrecords_name = "01-of-01%s.tfrecords" % suffix
        index_path = "01-of-01%s.index" % suffix

    if evaluate:
        dataset_type_name = "dev"
    elif predict:
        dataset_type_name = "test"
    else:
        dataset_type_name = "train"
    dirpath = os.path.join(args.data_dir, "tfrecords", dataset_type_name, args.label_type)
    if not os.path.exists(dirpath):
        dirpath = os.path.join(args.data_dir, "tfrecords", dataset_type_name)
    tfrecords_filepath = os.path.join(dirpath, tfrecords_name)
    if args.multi_tfrecords:
        index_pattern = os.path.join(dirpath, index_pattern)
        print("load tfrecords: %s, index: %s" % (tfrecords_filepath, index_pattern))
    else:
        index_path = os.path.join(dirpath, index_path)
        print("load tfrecords: %s, index: %s" % (tfrecords_filepath, index_path))
    label_list = load_labels(filepath=args.label_filepath, header=True)
    label_map = {label: i for i, label in enumerate(label_list)}

    # total sample number
    print("calc records num")
    if args.filename_pattern is not None and os.path.exists(os.path.join(args.data_dir, args.filename_pattern.format(dataset_type_name))):
        dataset_total_num = 0
        filepath = os.path.join(args.data_dir, args.filename_pattern.format(dataset_type_name))
        if filepath.endswith(".txt"):
            header = False
        elif filepath.endswith(".csv"):
            header = True
        protein_set = set()
        for row in file_reader(filepath, header=header, header_filter=True):
            protein_set.add(row[0])
        dataset_total_num = len(protein_set)
    else:
        if args.multi_tfrecords:
            dataset_total_num = sum(1 for _ in MultiTFRecordDataset(tfrecords_filepath, index_pattern=index_pattern, splits=None, description=None))
        else:
            dataset_total_num = sum(1 for _ in TFRecordDataset(tfrecords_filepath, index_path=index_path, description=None))
    print("records: %d" % dataset_total_num)
    # dataset for large data
    '''
    if predict or evaluate:
        shuffle_queue_size = None
    else:
        shuffle_queue_size = args.shuffle_queue_size
    '''
    shuffle_queue_size = args.shuffle_queue_size
    if args.multi_tfrecords:
        dataset = MultiTFRecordDataset(tfrecords_filepath, index_pattern=index_pattern, splits=None, description=None, shuffle_queue_size=shuffle_queue_size, transform=lambda x: parse_tfrecord(x,
                                                                                                                                                                                                 subword=subword,
                                                                                                                                                                                                 seq_tokenizer=seq_tokenizer,
                                                                                                                                                                                                 struct_tokenizer=struct_tokenizer,
                                                                                                                                                                                                 seq_max_length=args.seq_max_length,
                                                                                                                                                                                                 struct_max_length=args.struct_max_length,
                                                                                                                                                                                                 embedding_type=args.embedding_type,
                                                                                                                                                                                                 embedding_max_length=args.embedding_max_length,
                                                                                                                                                                                                 output_mode=args.output_mode,
                                                                                                                                                                                                 label_map=label_map,
                                                                                                                                                                                                 pad_on_left=False,
                                                                                                                                                                                                 pad_token=0,
                                                                                                                                                                                                 pad_token_segment_id=0,
                                                                                                                                                                                                 mask_padding_with_zero=True,
                                                                                                                                                                                                 trunc_type=args.trunc_type,
                                                                                                                                                                                                 cmap_type=args.cmap_type,
                                                                                                                                                                                                 cmap_thresh=args.cmap_thresh))
    else:
        dataset = TFRecordDataset(tfrecords_filepath, index_path=index_path, description=None, shuffle_queue_size=shuffle_queue_size,
                                  transform=lambda x: parse_tfrecord(x,
                                                                     subword=subword,
                                                                     seq_tokenizer=seq_tokenizer,
                                                                     struct_tokenizer=struct_tokenizer,
                                                                     seq_max_length=args.seq_max_length,
                                                                     struct_max_length=args.struct_max_length,
                                                                     embedding_type=args.embedding_type,
                                                                     embedding_max_length=args.embedding_max_length,
                                                                     output_mode=args.output_mode,
                                                                     label_map=label_map,
                                                                     pad_on_left=False,
                                                                     pad_token=0,
                                                                     pad_token_segment_id=0,
                                                                     mask_padding_with_zero=True,
                                                                     trunc_type=args.trunc_type,
                                                                     cmap_type=args.cmap_type,
                                                                     cmap_thresh=args.cmap_thresh))
    # return dataset and dataset size
    return dataset, dataset_total_num


if __name__ == "__main__":
    from tqdm import tqdm, trange
    data_dir = "/mnt/****/workspace/DeepProtFunc/dataset/rdrp_40_extend/protein/binary_class"
    tfrecords_name = "01-of-01_emb.tfrecords"
    index_name = "01-of-01_emb.index"
    tfrecords_filepath = os.path.join(data_dir, "tfrecords", "test", tfrecords_name)
    index_filepath = os.path.join(data_dir, "tfrecords", "test", index_name)
    def parse(x):
        '''
        embedding_len = x["emb_l"][0]
        seq = "".join([chr(v) for v in x["seq"]])
        print("seq len: %d" % len(seq))
        print("embedding_len: %d" % embedding_len)
        '''
        return torch.tensor(0, dtype=torch.long)

    dataset = TFRecordDataset(tfrecords_filepath, index_path=index_filepath, description=None, shuffle_queue_size=10000, transform=lambda x: parse(x))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    epoch_iterator = tqdm(loader, desc="Iteration", disable=False)
    cnt = 0
    for step, batch in enumerate(epoch_iterator):
        cnt += 1
    print("cnt: %d" % cnt)


