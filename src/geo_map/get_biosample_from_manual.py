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

parser = argparse.ArgumentParser(description='Extract BioSample metadata from Manual File.')

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

sra_2_biosample_manual = {}
manual = []
for idx in range(5):
    with open("./data/sra_manual_biosample_%02d_done.txt" % (idx + 1), "r") as rfp:
        reader = csv.reader(rfp)
        cnt = 0
        for row in reader:
            cnt += 1
            if cnt == 1:
                continue
            sra, url, biosample = row[0], row[1], row[2]
            biosample = biosample.strip()
            if "(" in biosample:
                biosample = biosample[0:biosample.index("(")].strip()
            elif "（" in biosample:
                biosample = biosample[0:biosample.index("（")].strip()
            sra_2_biosample_manual[sra] = biosample
            if sra in sra_query_list and (sra not in exists_sra_2_biosample or biosample not in exists_sra_2_biosample[sra]):
                manual.append([sra, biosample])

not_exists_biosample_sras = set(sra_query_list).difference(exists_sra_2_biosample.keys())
exist_manual_biosample_sras = set(sra_query_list).intersection(sra_2_biosample_manual.keys())
print("not_exists_biosample_sras size: %d" % len(not_exists_biosample_sras))
print("exist manual biosample size: %d" % len(exist_manual_biosample_sras))
print("manual len: %d" % len(manual))
unfound_biosample_sra_after_manual = not_exists_biosample_sras.difference(set(sra_2_biosample_manual.keys()))
print("unfound biosample sra after manual size: %d" % len(unfound_biosample_sra_after_manual))

exists_flag = False
if os.path.exists(args.output):
    exists_flag = True
with open(args.output, "a+") as wfp:
    writer = csv.writer(wfp)
    if not exists_flag:
        writer.writerow(["sra", "biosample"])
    for item in manual:
        writer.writerow(item)

with open("./data/unfound_biosample_sra_after_manual.txt", "w") as wfp:
    if unfound_biosample_sra_after_manual:
        for sra in unfound_biosample_sra_after_manual:
            wfp.write("%s\n" % sra)
print("-"*25 + "Get Biosample by Manual Done" + "-"*25)
