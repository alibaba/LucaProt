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
@email: sanyuan.**alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/2 11:50
@project: DeepProtFunc
@file: statistics
@desc: statistics the predicted results of three positive datasets, three negative datasets, and our checked RdRP dataset
'''
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default=None, type=str, required=True, help="dataset name")
parser.add_argument("--model_type", default=None, type=str, required=True, help="model type.")
parser.add_argument("--time_str", default=None, type=str, required=True, help="the running time string(yyyymmddHimiss) of model building.")
parser.add_argument("--step", default=None, type=str, required=True, help="the training global step of model finalization.")


dataset_list_list = [
    ["2022Cell_RdRP_with_pdb_emb", "2022NM_RdRP_with_pdb_emb", "2022Science_RdRP_with_pdb_emb"],
    ["All_RT_with_pdb_emb", "Eukaryota_DdRP_with_pdb_emb", "Eukaryota_RdRP_with_pdb_emb"],
    ["ours_checked_rdrp_final"]
]
save_filename_list = ["validation_positive_dataset.txt", "validation_negative_dataset.txt", "ours_checked_rdrp_final.txt"]
args = parser.parse_args()

save_dir = "../../predicts/%s/protein/binary_class/%s/%s/checkpoint-%s/" %(args.dataset_name, args.model_type, args.time_str, args.step)
for idx, dataset_list in enumerate(dataset_list_list):
    with open(os.path.join(save_dir, save_filename_list[idx]), "w") as wfp:
        statistics = {}
        totals = {}
        labels = set()
        for dataset in dataset_list:
            predict_result_filepath = "../../predicts/%s/protein/binary_class/%s/%s/checkpoint-%s/%s/pred_metrics.txt" % (args.dataset_name, args.model_type, args.time_str, args.step, dataset)
            statistics[dataset] = {}
            totals[dataset] = 0
            with open(predict_result_filepath, "r") as rfp:
                begin = False
                for line in rfp:
                    line = line.strip()
                    if line == "prediction statistics:":
                        begin = True
                    elif begin and "=" in line:
                        strs = line.split("=")
                        label = str(strs[0])
                        num = int(strs[1])
                        statistics[dataset][label] = num
                        totals[dataset] += num
                        labels.add(label)
        for dataset in dataset_list:
            for label in labels:
                if label not in statistics[dataset]:
                    statistics[dataset][label] = 0

        micro_avg = {}
        for item1 in statistics.items():
            dataset = item1[0]
            print("dataset: %s, total: %d" % (dataset, totals[dataset]))
            wfp.write("dataset: %s, total: %d\n" % (dataset, totals[dataset]))
            for item2 in item1[1].items():
                label = item2[0]
                num = item2[1]
                if label not in micro_avg:
                    micro_avg[label] = []
                rate = num/totals[dataset]
                micro_avg[label].append(rate)
                print("label: %s, num: %d, rate: %f(%d/%d)" %(label, num, rate, num, totals[dataset]))
                wfp.write("label: %s, num: %d, rate: %f(%d/%d)\n" %(label, num, rate, num, totals[dataset]))

        for item in micro_avg.items():
            print("label: %s %f(%s)" % (item[0], sum(item[1])/len(item[1]), str(item[1])))
            wfp.write("label: %s %f(%s)\n" % (item[0], sum(item[1])/len(item[1]), str(item[1])))
        print("-"*50)
        wfp.write("-"*50 + "\n")






