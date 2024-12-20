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
@datetime: 2022/12/30 15:13
@project: DeepProtFunc
@file: data_preprocess_into_tfrecords_for_rdrp
@desc: transform the dataset for model building into tfrecords
'''
import os
import random
import numpy as np
import multiprocessing
import tensorflow as tf
import sys, csv, torch
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import write_fasta, fasta_reader, clean_seq, load_labels, file_reader, common_amino_acid_set
except ImportError:
    from src.utils import write_fasta, fasta_reader, clean_seq, load_labels, file_reader, common_amino_acid_set


class GenerateTFRecord(object):
    def __init__(self, dataset_filename, structure_dir, embedding_dir, label_filepath, save_path, shuffle=False, num_shards=30):
        self.shuffle = shuffle
        self.dataset_filename = dataset_filename
        self.structure_dir = structure_dir
        self.embedding_dir = embedding_dir
        self.save_path = save_path
        self.num_shards = num_shards
        self.dataset = self.load_dataset(self.dataset_filename)
        self.prot_list = list(self.dataset.keys())
        if self.shuffle:
            for _ in range(5):
                random.shuffle(self.prot_list)
        self.label_filepath = label_filepath
        self.label_2_id = {label: idx for idx, label in enumerate(load_labels(self.label_filepath, header=True))}

        shard_size = (len(self.prot_list) + num_shards - 1)//num_shards
        indices = [(i * shard_size, (i + 1) * shard_size) for i in range(0, num_shards)]
        indices[-1] = (indices[-1][0], len(self.prot_list))
        self.indices = indices

    def load_dataset(self, header=True):
        '''
        load the dataset
        :param header: whether contains header in the dataset file
        :return:
        '''
        dataset = {}
        with open(self.dataset_filename, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1 and header:
                    continue
                # prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
                prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
                seq = seq.strip("*")
                dataset[prot_id] = [seq, None, None, label]
                if self.structure_dir:
                    structure_filepath = os.path.join(self.structure_dir, pdb_filename)
                    if os.path.exists(structure_filepath):
                        dataset[prot_id][1] = structure_filepath
                if self.embedding_dir:
                    embedding_filepath = os.path.join(self.embedding_dir, emb_filename)
                    if os.path.exists(embedding_filepath):
                        dataset[prot_id][2] = embedding_filepath
                    else:
                        embedding_filepath = os.path.join(self.embedding_dir.replace("_append", ""), emb_filename)
                        if os.path.exists(embedding_filepath):
                            dataset[prot_id][2] = embedding_filepath
                        else:
                            print("%s emb filepath not exists!" % prot_id)
        return dataset

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _dtype_feature(self):
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))

    def _int_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _serialize_example(self, obj_id, sequence, pdb_obj, embedding_obj, label):
        d_feature = {'id': self._bytes_feature(obj_id.encode()), 'seq': self._bytes_feature(sequence.encode()),
                     'L': self._int_feature(len(sequence))}

        cur_example_label = label
        if cur_example_label is None or len(cur_example_label) == 0:
            return None
        if isinstance(cur_example_label, list) or isinstance(cur_example_label, set):
            cur_example_label_ids = [self.label_2_id[v] for v in cur_example_label]
            d_feature['label'] = self._int_feature(cur_example_label_ids)
        else:
            cur_example_label_id = self.label_2_id[cur_example_label]
            d_feature['label'] = self._int_feature(cur_example_label_id)

        if embedding_obj:
            d_feature['emb_l'] = self._int_feature(embedding_obj["L"][1])
            d_feature['emb_size'] = self._int_feature(embedding_obj["d"][1])
            for item in embedding_obj.items():
                name = item[0]
                dtype = item[1][0]
                value = item[1][1]
                if isinstance(value, np.ndarray):
                    value = list(value.reshape(-1))
                elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                    value = [value]
                if dtype == "str":
                    d_feature[name] = self._bytes_feature(value)
                elif dtype == "int":
                    d_feature[name] = self._int_feature(value)
                else:
                    d_feature[name] = self._float_feature(value)
        if pdb_obj:
            d_feature['pdb_l'] = self._int_feature(pdb_obj["L"][1])
            for item in pdb_obj.items():
                name = item[0]
                dtype = item[1][0]
                value = item[1][1]
                if isinstance(value, np.ndarray):
                    value = list(value.reshape(-1))
                elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                    value = [value]
                if dtype == "str":
                    d_feature[name] = self._bytes_feature(value)
                elif dtype == "int":
                    d_feature[name] = self._int_feature(value)
                else:
                    d_feature[name] = self._float_feature(value)
        example = tf.train.Example(features=tf.train.Features(feature=d_feature))
        return example.SerializeToString()

    def _convert_numpy_folder(self, idx):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tfrecord_fn = os.path.join(self.save_path, '%0.2d-of-%0.2d%s.tfrecords' % (idx + 1, self.num_shards, "_pdb_emb" if self.structure_dir and self.embedding_dir else ("_pdb" if self.structure_dir else "_emb" if self.embedding_dir else "")))
        # writer = tf.python_io.TFRecordWriter(tfrecord_fn)
        print("Save path:", tfrecord_fn)
        writer = tf.io.TFRecordWriter(tfrecord_fn)
        tmp_prot_list = self.prot_list[self.indices[idx][0]:self.indices[idx][1]]
        print("Serializing %d examples into %s" % (len(tmp_prot_list), tfrecord_fn))

        for i, protein_id in enumerate(tmp_prot_list):
            if i % 500 == 0:
                print("### Iter = %d/%d" % (i, len(tmp_prot_list)))
            item = self.dataset[protein_id]
            protein_seq = item[0]
            protein_label = item[-1]

            pdb_obj = None
            if item[1]:
                pdb_file = item[1]
                cmap = np.load(pdb_file, allow_pickle=True)
                ca_dist_matrix = cmap['C_alpha']
                cb_dist_matrix = cmap['C_beta']
                assert protein_seq == str(cmap['seqres'].item())
                assert protein_label == cmap['label'].item()
                pdb_obj = {
                    "L": ["int", ca_dist_matrix.shape[0]],
                    "C_alpha_dist_matrix": ["float", ca_dist_matrix],
                    "C_beta_dist_matrix": ["float", cb_dist_matrix]
                }
            embedding_obj = None
            if item[2]:
                embedding_file = item[2]
                embedding_obj = torch.load(embedding_file)
                # embeding_size
                bos_representations = embedding_obj["bos_representations"][36].numpy()
                # L * embeding_size
                representations = embedding_obj["representations"][36].numpy()
                # L * L
                contacts = embedding_obj["contacts"].numpy()
                if clean_seq(protein_id, protein_seq) != embedding_obj["seq"]:
                    print(set(protein_seq).difference(set(embedding_obj["seq"])))
                    print(set(embedding_obj["seq"]).difference(set(protein_seq)))

                embedding_obj = {
                    "L": ["int", representations.shape[0]], # representations.shape[0]
                    "d": ["int", representations.shape[1]],
                    "bos_representations": ["float", bos_representations],
                    "representations": ["float", representations],
                    # "contacts": ["float", contacts]
                }
            example = self._serialize_example(protein_id, protein_seq, pdb_obj, embedding_obj, protein_label)
            if example is None:
                continue
            writer.write(example)

        print("label size: %d" % len(self.label_2_id))
        print("Writing {} done!".format(tfrecord_fn))

    def run(self, num_threads):
        pool = multiprocessing.Pool(processes=num_threads)
        shards = [idx for idx in range(0, self.num_shards)]
        pool.map(self._convert_numpy_folder, shards)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name",
                    default="rdrp_extend_40",
                    required=True,
                    type=str,
                    help="transform into tfrecords"
                    )
parser.add_argument("--train", action="store_true", help="the dataset type")
args = parser.parse_args()

if __name__ == "__main__":
    if args.train:
        dataset_type_list = ["train"]
    else:
        dataset_type_list = ["train", "dev", "test"]
    num_shards = [1, 1, 1]
    for idx, dataset_type in enumerate(dataset_type_list):
        dataset_filename = "../dataset/%s/protein/binary_class/%s_with_pdb_emb.csv" % (args.dataset_name, dataset_type)
        structure_dir = None
        embedding_dir = "../dataset/%s/protein/binary_class/embs/" % args.dataset_name
        label_filepath = "../dataset/%s/protein/binary_class/label.txt" % args.dataset_name
        save_path = "../dataset/%s/protein/binary_class/tfrecords/%s/" % (args.dataset_name, dataset_type)
        tfr = GenerateTFRecord(dataset_filename,
                               structure_dir,
                               embedding_dir, label_filepath, save_path, shuffle=True if dataset_type == "train" else False, num_shards=num_shards[idx])
        tfr.run(num_threads=1)
    '''
    Note: need to build the index file, the cmd: python -m tfrecord.tools.tfrecord2idx 01-of-01_emb.tfrecords 01-of-01_emb.index
    '''
    try:
        from utils import file_reader, common_amino_acid_set
    except ImportError:
        from src.utils import file_reader, common_amino_acid_set
    not_common_seqs = []
    total = 0
    not_common = 0
    dataset_type_list = ["train", "dev", "test"]
    with open("../dataset/%s/protein/binary_class/contain_not_common_amino_acid.fasta" % args.dataset_name, "w") as wfp:
        for dataset_type in dataset_type_list:
            dataset_filename = "../dataset/%s/protein/binary_class/%s_with_pdb_emb.csv" % (args.dataset_name, dataset_type)
            for row in file_reader(dataset_filename, header=True, header_filter=True):
                prot_id, seq, seq_len, pdb_filename, ptm, mean_plddt, emb_filename, label, source = row
                diff = set(list(seq)).difference(common_amino_acid_set)
                total += 1
                if len(diff) > 0:
                    not_common += 1
                    wfp.write(prot_id + "\n")
                    wfp.write(seq + "\n")
                    wfp.write(str(diff) + "\n")
    print("%d, %d" % (total, not_common))








