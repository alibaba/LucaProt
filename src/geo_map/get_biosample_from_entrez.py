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
@datetime: 2023/3/14 16:27
@project: DeepProtFunc
@file: get_biosample_from_entrez
@desc: use NCBI Entrez to get biosample id according to SRA ID
'''
import os, csv
import argparse
from Bio import Entrez
import xml.etree.ElementTree as ET
import pprint

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser(description='Extract BioSample metadata from NCBI Entrez.')

parser.add_argument('-i', '--input',  metavar=("in"), required=True,
                    type=str, help='Input file of sra list')

parser.add_argument('-o', '--output', metavar=("out"), required=True,
                    type=str, help='Output filename presenting lookup results')

parser.add_argument('-e', '--email', required=True,
                    type=str, help='Input your email address for Entrez queries')

parser.add_argument('-idx', '--sra_idx', required=True,
                    type=int, help='sra id col index in sra file')

parser.add_argument('-u', '--update', action='store_true',
                    help='Update existed results')

parser.add_argument('--debug', action='store_true',
                    help='Debug this running')
args = parser.parse_args()


def indent(elem, level=0):
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem

Entrez.email = args.email

header = None
sra_2_biosample = {}
if os.path.exists(args.output):
    with open(args.output, "r") as rfp:
        reader = csv.reader(rfp, delimiter=',')
        header = next(reader)
        for row in reader:
            sra_id = row[0]
            biosample_id = row[1]
            sra_2_biosample[sra_id] = set([biosample_id])
print("exists biosample sra size: %d" % len(sra_2_biosample))

sra_query_list = []
with open(args.input, 'r') as rfp:
    reader = csv.reader(rfp, delimiter='\t')
    next(reader)
    for row in reader:
        sra = row[args.sra_idx]
        if sra in sra_2_biosample:
            if not args.update:
                if args.debug:
                    print("the biosample id(%s) of sra: %s exists." % (sra, sra_2_biosample[sra]))
            else:
                sra_query_list.append(sra)
        else:
            sra_query_list.append(sra)


cur_query = set(sra_query_list)
print("sra want to convect to biosample size: %d" % len(cur_query))
run_times = 0
changed = True
while changed:
    run_times += 1
    print("run times: %d" % run_times)
    cnt = 0
    changed = False
    cur_done_query = set()
    for sra_id in cur_query:
        try:
            handle = None
            if sra_id.startswith("SAM"):
                print("skipping %s as it looks like a biosample." % sra_id)
            else:
                try:
                    handle = Entrez.efetch(db="sra", id=sra_id)
                except Exception as e:
                    print(e)
                    print("skipping %s, Entrez error" % sra_id)
                    handle = None
            if handle:
                tree = ET.parse(handle)
                root = tree.getroot()
                for sd in root.iter('SAMPLE'):
                    if args.debug:
                        indent(sd)
                        ET.dump(sd)
                    accession = None
                    if "accession" in sd.attrib:
                        accession = sd.attrib["accession"].strip()
                        if args.debug:
                            print("SRA BioSample Accession is %s" % accession)
                    alias = None
                    if "alias" in sd.attrib:
                        alias = sd.attrib["alias"].strip()
                        if args.debug:
                            print("SRA BioSample Accession Alias is %s " % alias)
                    flag = False
                    if accession and len(accession) > 0:
                        flag = True
                        if sra_id not in sra_2_biosample:
                            sra_2_biosample[sra_id] = set()
                        sra_2_biosample[sra_id].add(accession.strip())
                    if alias and len(alias) > 0:
                        flag = True
                        if sra_id not in sra_2_biosample:
                            sra_2_biosample[sra_id] = set()
                        sra_2_biosample[sra_id].add(alias.strip())

                    if flag:
                        changed = True
                        cur_done_query.add(sra_id)
                    if args.debug:
                        print("{}, {} => {}".format(accession, alias, sra_id))
        except Exception as e:
            print("sra to biosample exception: %s " % sra_id)
            print(e)
        cnt += 1
        if cnt % 1000 == 0:
            print("done: %d, rate: %f" % (len(cur_done_query), len(cur_done_query)/len(cur_query)))
    cur_query = cur_query.difference(cur_done_query)
    print("cur time done: %d, not done cnt: %d" % (len(cur_done_query), len(cur_query)))

print("total done: %d, not done cnt: %d" % (len(sra_2_biosample), len(cur_query)))
with open(args.output, "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["sra", "biosample"])
    for item in sra_2_biosample.items():
        for v in item[1]:
            writer.writerow([item[0], v])


with open("./data/unfound_biosample_sra_after_entrez.txt", "w") as wfp:
    if cur_query:
        for sra in cur_query:
            wfp.write("%s\n" % sra)
print("-"*25 + "Get Biosample by Entrez Done" + "-"*25)

