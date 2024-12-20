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
@datetime: 2022/12/25 18:50
@project: DeepProtFunc
@file: verify_train_dataset
@desc: verify the training set
'''

import os, csv, torch
reader = csv.reader(
    open("../../dataset/rdrp_40_extend/protein/binary_class/train_with_pdb_emb.csv")
)
statistics = {}
cnt = 0
for row in reader:
    cnt += 1
    if cnt == 1:
        continue
    prot_id,seq,seq_len,pdb_filename,ptm,mean_plddt,emb_filename,label,source = row
    emb_filepath = os.path.join(
        "../../dataset/rdrp_40_extend/protein/binary_class/embs",
        emb_filename
    )
    if not os.path.exists(emb_filepath):
        emb_filepath = os.path.join(
            "../../dataset/rdrp_40/protein/binary_class/embs",
            emb_filename
        )
    embedding = torch.load(emb_filepath)
    assert prot_id == embedding["protein_id"]
    label = int(label)
    if label in statistics:
        statistics[label] += 1
    else:
        statistics[label] = 1
print(statistics)
print(statistics[0]/statistics[1])
'''
{0: 229434, 1: 5980}
38.366889632107025
'''