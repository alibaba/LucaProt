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
@datetime: 2023/3/15 11:14
@project: DeepProtFunc
@file: test
@desc: xxxx
'''
import os, csv
import argparse
from Bio import Entrez
import xml.etree.ElementTree as ET

'''
Entrez.email = "12657866q@qq.com"

sra_id = "SRR7523395"

handle = Entrez.efetch(db="sra", id=sra_id)
# handle = Entrez.esearch(db="sra", id="DRR110568", retmax=50)
if handle:
    tree = ET.parse(handle)
    root = tree.getroot()
    print(root)
    for sd in root.iter('SAMPLE'):
        print(sd)
    print("------------")
    accession = None
    if "accession" in sd.attrib:
        accession = sd.attrib["accession"].strip()
        print("SRA BioSample Accession is %s" % accession)
    alias = None
    if "alias" in sd.attrib:
        alias = sd.attrib["alias"].strip()
        print("SRA BioSample Accession Alias is %s " % alias)
'''

filepath = "data/all_sra_lat_lon.csv"
sra_lat_lon_info1 = {}
lat_lon_sra1 = {}
sra_set1 = set()
with open(filepath, "r") as rfp:
    cnt = 0
    reader = csv.reader(rfp)
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        sra, name, lat, lon, data_type = row[0], row[1], float(row[2]), float(row[3]), row[4]
        sra_lat_lon_info1[sra] = [lat, lon]
        lat_lon = "%0.8f###%0.8f" %(lat, lon)
        if lat_lon not in lat_lon_sra1:
            lat_lon_sra1[lat_lon] = set()
        lat_lon_sra1[lat_lon].add(sra)
        sra_set1.add(sra)
print("sra size: %d, lat_lon_sra size: %d" % (len(sra_set1), len(lat_lon_sra1)))

filepath = "data/all_sra_lat_lon_1.csv"
sra_lat_lon_info2 = {}
lat_lon_sra2 = {}
sra_set2 = set()
with open(filepath, "r") as rfp:
    cnt = 0
    reader = csv.reader(rfp)
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        sra, name, lat, lon, data_type = row[0], row[1], float(row[2]), float(row[3]), row[4]
        sra_lat_lon_info2[sra] = [lat, lon]
        lat_lon = "%0.8f###%0.8f" %(lat, lon)
        if lat_lon not in lat_lon_sra2:
            lat_lon_sra2[lat_lon] = set()
        lat_lon_sra2[lat_lon].add(sra)
        sra_set2.add(sra)
print("sra size: %d, lat_lon_sra size: %d" % (len(sra_set2), len(lat_lon_sra2)))

diff = set(lat_lon_sra2.keys()).difference(lat_lon_sra1.keys())
for item in set(lat_lon_sra2.keys()).difference(lat_lon_sra1.keys()):
    print(lat_lon_sra2[item])