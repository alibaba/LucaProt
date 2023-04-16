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
@datetime: 2023/3/11 20:09
@project: DeepProtFunc
@file: get_biosample_from_entrez
@desc: update biosample id from data/00all_SraRunInfo_new.csv
'''
import csv, argparse
import os.path


parser = argparse.ArgumentParser(description='Extract BioSample metadata from Updated File.')

parser.add_argument('-i', '--input',  metavar=("in"), required=True,
                    type=str, help='Input file of sra list')

parser.add_argument('-o', '--output', metavar=("out"), required=True,
                    type=str, help='Output filename presenting lookup results')

parser.add_argument('-idx', '--sra_idx', required=True,
                    type=int, help='sra id col index in sra file')

args = parser.parse_args()

sra_query_list = []
with open(args.input, 'r') as rfp:
    reader = csv.reader(rfp, delimiter='\t')
    next(reader)
    for row in reader:
        sra = row[args.sra_idx]
        sra_query_list.append(sra)


exists_sra_2_biosample = {}
if os.path.exists(args.output):
    with open(args.output, "r") as rfp:
        reader = csv.reader(rfp, delimiter=',')
        header = next(reader)
        for row in reader:
            sra_id = row[0]
            biosample_id = row[1]
            exists_sra_2_biosample[sra_id] = set([biosample_id])

print("exists biosample sra size: %d" % len(exists_sra_2_biosample))

sra_2_biosample_update = {}
updated = []
with open("./data/00all_SraRunInfo_new.csv", "r") as rfp:
    reader = csv.reader(rfp)
    BioSample_Idx = 0
    BioProject_Idx = 0
    Run_Idx = 0
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt == 1:
            BioSample_Idx = row.index("BioSample")
            BioProject_Idx = row.index("BioProject")
            Run_Idx = row.index("Run")
            continue
        bioSample = row[BioSample_Idx]
        sra = row[Run_Idx]
        sra_2_biosample_update[sra] = bioSample
        if sra in sra_query_list and (sra not in exists_sra_2_biosample or bioSample not in exists_sra_2_biosample[sra]):
            updated.append([sra, bioSample])

not_exists_biosample_sras = set(sra_query_list).difference(exists_sra_2_biosample.keys())
exist_updated_biosample_sras = set(sra_query_list).intersection(sra_2_biosample_update.keys())
print("not_exists_biosample_sras size: %d" % len(not_exists_biosample_sras))
print("exist updated biosample size: %d" % len(exist_updated_biosample_sras))
print("updated len: %d" % len(updated))
unfound_biosample_sra_after_update = not_exists_biosample_sras.difference(set(sra_2_biosample_update.keys()))
print("unfound biosample sra after update size: %d" % len(unfound_biosample_sra_after_update))

exists_flag = False
if os.path.exists(args.output):
    exists_flag = True
with open(args.output, "a+") as wfp:
    writer = csv.writer(wfp)
    if not exists_flag:
        writer.writerow(["sra", "biosample"])
    for item in updated:
        writer.writerow(item)

with open("./data/unfound_biosample_sra_after_update.txt", "w") as wfp:
    if unfound_biosample_sra_after_update:
        for sra in unfound_biosample_sra_after_update:
            wfp.write("%s\n" % sra)
print("-"*25 + "Get Biosample by Updating Done" + "-"*25)