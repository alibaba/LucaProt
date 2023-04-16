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
@datetime: 2023/3/14 18:42
@project: DeepProtFunc
@file: download_biosample_page
@desc: download biosample page info
'''
import os.path
import time
import requests
import csv
import argparse


def download_content(url):
    response = requests.get(url).text
    return response


def save_to_file(filename, content):
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(content)


def main(inputs):
    if not os.path.exists("./sra_biosample_html"):
        os.makedirs("./sra_biosample_html")
    for row in inputs:
        sra, bioSample = row[0], row[1]
        url = "https://www.ncbi.nlm.nih.gov/biosample/%s" % bioSample
        result = download_content(url)
        save_to_file("./sra_biosample_html/%s_%s.html" % (sra, bioSample.replace("/", "_")), result)
        time.sleep(1)

parser = argparse.ArgumentParser(description='Extract BioSample metadata from NCBI Entrez.')
parser.add_argument('-i', '--input',  metavar=("in"), required=True,
                    type=str, help='Input file of sra to biosample list')
args = parser.parse_args()

inputs = []
with open(args.input, "r") as rfp:
    reader = csv.reader(rfp)
    next(reader)
    for row in reader:
        sra, biosample = row[0], row[1]
        inputs.append([sra, biosample])

# download
main(inputs)

print("-"*25 + "Download Page Done" + "-"*25)