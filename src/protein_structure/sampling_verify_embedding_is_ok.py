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
@datetime: 2023/2/10 10:14
@project: DeepProtFunc
@file: sampling_verify_embedding_is_ok
@desc: verify embedding file is ok(sampling)
'''
import csv, random
import argparse, torch
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", default=None, required=True, type=str, help="")
parser.add_argument("--emb_dir", default=None, required=True,  type=str, help="")
parser.add_argument("--sampling_rate", default=None, required=True, type=float, help="sampling rate.")
args = parser.parse_args()
with open(args.filepath, "r") as rfp:
    reader = csv.reader(rfp)
    cnt = 0
    done_num = 0
    for row in reader:
        cnt += 1
        if cnt == 1 or random.random() < args.sampling_rate:
            continue
        prot_id = row[0]
        emb_filename = row[6]
        try:
            protein_id = torch.load(os.path.join(args.emb_dir, emb_filename))["protein_id"]
            assert protein_id == prot_id
        except Exception as e:
            print(e)
            print(protein_id == prot_id)
            raise Exception("error")
        done_num += 1
