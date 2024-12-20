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
@datetime: 2022/12/25 16:50
@project: DeepProtFunc
@file: tf_records_generator
@desc: transform the dataset for model building into tfrecords
'''
import os, sys
import argparse
import tensorflow as tf
import multiprocessing
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import *
except ImportError:
    from src.utils import *


class GenerateTFRecord(object):
    def __init__(
            self,
            prot_list,
            label_2_id,
            npz_dir,
            save_path,
            label_type,
            num_shards=30
    ):
        self.prot_list = prot_list
        self.label_2_id = label_2_id
        self.npz_dir = npz_dir
        self.save_path = save_path
        self.num_shards = num_shards
        self.label_type = label_type
        shard_size = len(prot_list)//num_shards
        indices = [(i * shard_size, (i+1) * shard_size) for i in range(0, num_shards)]
        indices[-1] = (indices[-1][0], len(prot_list))
        self.indices = indices
        self.labels = set()

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

    def _serialize_example(self, obj_id, sequence, obj, label):
        d_feature = {}
        d_feature['id'] = self._bytes_feature(obj_id.encode())
        d_feature['seq'] = self._bytes_feature(sequence.encode())
        d_feature['L'] = self._int_feature(len(sequence))
        cur_example_label = label
        if isinstance(cur_example_label, dict):
            assert self.label_type is not None
            cur_example_label = cur_example_label[self.label_type]
        if cur_example_label is None or len(cur_example_label) == 0:
            return None
        if isinstance(cur_example_label, list) or isinstance(cur_example_label, set):
            self.labels = self.labels.union(set(cur_example_label))
            cur_example_label_ids = [self.label_2_id[v] for v in cur_example_label]
            d_feature['label'] = self._int_feature(cur_example_label_ids)
        else:
            self.labels.add(cur_example_label)
            cur_example_label_id = self.label_2_id[cur_example_label]
            d_feature['label'] = self._int_feature(cur_example_label_id)

        for item in obj.items():
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
        tfrecord_fn = os.path.join(self.save_path, '%0.2d-of-%0.2d.tfrecords' % (idx, self.num_shards))
        # writer = tf.python_io.TFRecordWriter(tfrecord_fn)
        writer = tf.io.TFRecordWriter(tfrecord_fn)
        print("### Serializing %d examples into %s" % (len(self.prot_list), tfrecord_fn))

        tmp_prot_list = self.prot_list[self.indices[idx][0]:self.indices[idx][1]]

        for i, prot in enumerate(tmp_prot_list):
            if i % 500 == 0:
                print("### Iter = %d/%d" % (i, len(tmp_prot_list)))
            pdb_file = os.path.join(self.npz_dir, prot + '.npz')
            if os.path.isfile(pdb_file):
                cmap = np.load(pdb_file, allow_pickle=True)
                sequence = str(cmap['seqres'].item())
                ca_dist_matrix = cmap['C_alpha']
                cb_dist_matrix = cmap['C_beta']
                label = cmap['label'].item()
                example = self._serialize_example(prot, sequence, {
                    "L": ["int", ca_dist_matrix.shape[0]],
                    "C_alpha_dist_matrix": ["float", ca_dist_matrix],
                    "C_beta_dist_matrix": ["float", cb_dist_matrix]
                }, label)
                if example is None:
                    continue
                writer.write(example)
            else:
                print(pdb_file, " not exists.")
        print("label size: %d" % len(self.labels))
        print("Writing {} done!".format(tfrecord_fn))

    def run(self, num_threads):
        pool = multiprocessing.Pool(processes=num_threads)
        shards = [idx for idx in range(0, self.num_shards)]
        pool.map(self._convert_numpy_folder, shards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_type',
                        type=str,
                        default='train',
                        choices=['train', 'dev', 'test'],
                        help="dataset filepath"
                        )
    parser.add_argument(
        '--label_type',
        type=str,
        default='molecular_function',
        choices=['molecular_function', 'biological_process', 'cellular_component'],
        help="label type"
    )
    parser.add_argument(
        '--label_path',
        type=str,
        default=None,
        help="label filepath"
    )
    parser.add_argument(
        '--prot_id_path',
        type=str,
        default=None,
        help="Input file (*.txt) with a set of protein IDs with distMAps in npz_dir."
    )
    parser.add_argument(
        '--npz_dir',
        type=str,
        default=None,
        help="Directory with distance maps saved in *.npz format to be loaded."
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=20,
        help="Number of threads (CPUs) to use in the computation."
    )
    parser.add_argument(
        '--num_shards',
        type=int,
        default=20,
        help="Number of tfrecord files per protein set."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help="Directory with tfrecord files for model training."
    )
    args = parser.parse_args()

    print("#" * 50)
    print(str(args))
    print("#" * 50)

    prot_id_list = [v for v in file_reader(args.prot_id_path, header=False, header_filter=True)]
    label_2_id = {v: idx for idx, v in enumerate(load_labels(args.label_path, header=True))}

    args.save_path = os.path.join(args.save_path, args.dataset_type, args.label_type)
    tfr = GenerateTFRecord(
        prot_id_list,
        label_2_id,
        args.npz_dir,
        args.save_path,
        args.label_type,
        num_shards=args.num_shards
    )
    tfr.run(
        num_threads=args.num_threads
    )
    print("%s %s label size: %d" % (
        args.dataset_type,
        args.label_type,
        len(tfr.labels)
    ))

    # python tf_records_generator.py --dataset_type train --label_type biological_process --label_path ../dataset/go/protein/multi_label/label_biological_process.txt --prot_id_path ../dataset/go/protein/multi_label/train.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1
    # python tf_records_generator.py --dataset_type dev --label_type biological_process --label_path ../dataset/go/protein/multi_label/label_biological_process.txt --prot_id_path ../dataset/go/protein/multi_label/dev.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1
    # python tf_records_generator.py --dataset_type test --label_type biological_process --label_path ../dataset/go/protein/multi_label/label_biological_process.txt --prot_id_path ../dataset/go/protein/multi_label/test.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1

    # python tf_records_generator.py --dataset_type train --label_type molecular_function --label_path ../dataset/go/protein/multi_label/label_molecular_function.txt --prot_id_path ../dataset/go/protein/multi_label/train.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1
    # python tf_records_generator.py --dataset_type dev --label_type molecular_function --label_path ../dataset/go/protein/multi_label/label_molecular_function.txt --prot_id_path ../dataset/go/protein/multi_label/dev.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1
    # python tf_records_generator.py --dataset_type test --label_type molecular_function --label_path ../dataset/go/protein/multi_label/label_molecular_function.txt --prot_id_path ../dataset/go/protein/multi_label/test.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1

    # python tf_records_generator.py --dataset_type train --label_type cellular_component --label_path ../dataset/go/protein/multi_label/label_cellular_component.txt --prot_id_path ../dataset/go/protein/multi_label/train.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1
    # python tf_records_generator.py --dataset_type dev --label_type cellular_component --label_path ../dataset/go/protein/multi_label/label_cellular_component.txt --prot_id_path ../dataset/go/protein/multi_label/dev.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1
    # python tf_records_generator.py --dataset_type test --label_type cellular_component --label_path ../dataset/go/protein/multi_label/label_cellular_component.txt --prot_id_path ../dataset/go/protein/multi_label/test.txt --npz_dir ../dataset/go/protein/multi_label/npz/ --save_path ../dataset/go/protein/multi_label/tfrecords --num_shards 1 --num_threads 1