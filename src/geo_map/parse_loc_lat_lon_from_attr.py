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
@datetime: 2023/3/14 18:52
@project: DeepProtFunc
@file: extract_attr_from_biosample_page.py
@desc: extract attributes from biosample page
'''
import os, json, csv
import pandas as pd
import argparse

import re
lat_lon_pattern1 = "[(]lat.*(-*[.\d]+).*lon.*(-*[.\d]+).*[)]"


def extract_lat_lon(s):
    return re.search(lat_lon_pattern1, s)


def is_none(s):
    '''
    judge the input is null
    :param s:
    :return:
    '''
    if s is None:
        return True
    s = str(s)
    s = s.strip()
    if len(s) == 0:
        return True
    s = s.lower()
    if s in ["none", "n", "null", "not applicable", "nil", "nan", "na", "not provided", "not collected", "missing"]:
        return True
    return False

parser = argparse.ArgumentParser(description='Parse Lat Lon info fromn attr.')

parser.add_argument('-i', '--input',  metavar=("in"), required=True,
                    type=str, help='Input file of sra list')

parser.add_argument('-idx', '--sra_idx', required=True,
                    type=int, help='sra id col index in sra file')

args = parser.parse_args()


sra_set = set()
with open(args.input, 'r') as rfp:
    reader = csv.reader(rfp, delimiter='\t')
    next(reader)
    for row in reader:
        sra = row[args.sra_idx]
        sra_set.add(sra)


print("len: %d" % len(sra_set))

biosample_set = set()

exists_html_sra_list = set()
done_sra_stats = {}
done_sra_set = set()
sra_2_biosample = {}
with open("./data/00all_SRA_run_res_simple_all.csv", "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["SRA_Run", "biosample", "geo_loc_name", "marine_region", "geographic_location(region_and_locality)",
                     "lat_lon", "geographic_location(latitude)", "geographic_location(longitude)", "geographic_location(depth)",
                     "latitude_start", "latitude_end", "longitude_start", "longitude_end", "submission"])
    for filename in os.listdir("./data"):
        if "_res.txt" in filename:
            print("filename: %s" % filename)
            with open("./data/%s" % filename, "r") as rfp:
                for line in rfp:
                    obj = json.loads(line.strip())
                    new_obj = {}
                    for key in obj.keys():
                        new_obj[key.lower()] = obj[key]
                    obj = new_obj
                    sra_info = obj["sra"]
                    if isinstance(sra_info, str) and "[" in sra_info:
                        sra_info = eval(sra_info)

                    biosample = obj["biosample"]

                    tmp = set()
                    if isinstance(sra_info, list):
                        for item in sra_info:
                            if isinstance(item, list):
                                for v in item:
                                    tmp.add(v)
                            else:
                                tmp.add(item)
                    else:
                        tmp.add(sra_info)
                    if len(tmp) == 0:
                        print("not sra:", line)
                    for sra in tmp:
                        if sra not in sra_2_biosample:
                            sra_2_biosample[sra] = []
                        sra_2_biosample[sra].append(biosample)
                        exists_html_sra_list.add(sra)
                        cur_lat_lon = None
                        if "description" in obj:
                            description = obj["description"].lower()
                            idx = description.find("position:")
                            if idx > -1:
                                end_idx = description[idx:].find('";')
                                cur_lat_lon = description[idx+len("position:"):end_idx+idx]
                            else:
                                cur_lat_lon = extract_lat_lon(description)
                                if cur_lat_lon:
                                    cur_lat_lon = cur_lat_lon.group()
                                    idx = cur_lat_lon.find(")")
                                    if idx > 0:
                                        cur_lat_lon = cur_lat_lon[0:idx+1]

                        attr = obj["attributes"]

                        if not isinstance(attr, str):
                            new_attr = {}
                            for key in attr.keys():
                                new_attr[key.lower()] = attr[key]
                            attr = new_attr
                        geo_loc_name = None
                        if "geo_loc_name" in attr:
                            geo_loc_name = attr["geo_loc_name"]

                        marine_region = None
                        if "marine region" in attr:
                            marine_region = attr["marine region"]

                        geographic_location = None
                        if "geographic location (region and locality)" in attr:
                            geographic_location = attr["geographic location (region and locality)"]
                        elif "geographic location" in attr:
                            # China:Shangrao 28.54N 118.03E
                            geographic_location = attr["geographic location"]
                            # print(geographic_location)

                        lat_lon = None
                        if "lat_lon" in attr:
                            lat_lon = attr["lat_lon"]
                        if "latitude and longitude" in attr:
                            lat_lon = attr["latitude and longitude"]
                        if is_none(lat_lon):
                            lat_lon = cur_lat_lon

                        geographic_location_lat = None
                        if "geographic location (latitude)" in attr:
                            geographic_location_lat = attr["geographic location (latitude)"]

                        geographic_location_lon = None
                        if "geographic location (longitude)" in attr:
                            geographic_location_lon = attr["geographic location (longitude)"]

                        geographic_location_depth = None
                        if "geographic location (depth)" in attr:
                            geographic_location_depth = attr["geographic location (depth)"]

                        latitude_start = None
                        if "latitude start" in attr:
                            latitude_start = attr["latitude start"]

                        latitude_end = None
                        if "latitude end" in attr:
                            latitude_end = attr["latitude end"]

                        longitude_start = None
                        if "longitude start" in attr:
                            longitude_start = attr["longitude start"]

                        longitude_end = None
                        if "longitude end" in attr:
                            longitude_end = attr["longitude end"]

                        center_name = None
                        if not isinstance(attr, str):
                            for key in attr.items():
                                if "center" in key and "name" in key:
                                    center_name = attr[key]
                            if is_none(center_name):
                                for key in attr.items():
                                    if "center" in key:
                                        center_name = attr[key]
                        if is_none(center_name) and "submission" in obj:
                            center_name = obj["submission"]

                        new_row = [sra, biosample,
                                   geo_loc_name, marine_region, geographic_location,
                                   lat_lon, geographic_location_lat, geographic_location_lon, geographic_location_depth,
                                   latitude_start, latitude_end, longitude_start, longitude_end, center_name]
                        none_col_num = 0
                        if is_none(geo_loc_name):
                            none_col_num += 1
                        if is_none(marine_region):
                            none_col_num += 1
                        if is_none(geographic_location):
                            none_col_num += 1
                        if is_none(lat_lon):
                            none_col_num += 1
                        if is_none(geographic_location_lat):
                            none_col_num += 1
                        if is_none(geographic_location_lon):
                            none_col_num += 1
                        if is_none(latitude_start):
                            none_col_num += 1
                        if is_none(latitude_end):
                            none_col_num += 1
                        if is_none(longitude_start):
                            none_col_num += 1
                        if is_none(longitude_end):
                            none_col_num += 1
                        if none_col_num == 10 and is_none(center_name):
                            o = json.loads(line)
                            del o["title"]
                            del o["Identifiers"]
                            del o["Organism"]
                            if "Description" in o:
                                print(sra, biosample)
                                print(o["Description"])
                            else:
                                print(sra, biosample)
                                print(o)
                            print(line)
                            continue
                        biosample_set.add(biosample)
                        done_sra_set.add(sra)
                        writer.writerow(new_row)

                        if sra in done_sra_stats:
                            done_sra_stats[sra].append([new_row, none_col_num])
                        else:
                            done_sra_stats[sra] = [[new_row, none_col_num]]
print("biosample_set size: %d" % len(biosample_set))
# the SRAs with no html file because them have a new biosample, and the biosample is obtained from data/00all_SraRunInfo_new.csv
not_exists_html_sras = sra_set.difference(exists_html_sra_list)
print("not_exists_html_sras: %d" % len(not_exists_html_sras))
print(not_exists_html_sras)


undone_sra_set = sra_set.difference(done_sra_set)
print("undone_sra_set: ")
print(undone_sra_set)
with open("./data/00all_SRA_sra_undo.txt", "w") as wfp:
    for item in undone_sra_set:
        wfp.write("%s\n" % item.strip())
'''

for item in sra_2_biosample.items():
    if len(set(item[1])) > 1:
        print(item)
'''
not_exist_sra_set = done_sra_set.difference(sra_set)
print("not_exist_sra_set: ")
print(not_exist_sra_set)

print("total: %d, done: %d, undo_sra_set: %d, not_exist_sra_set: %d " % (len(sra_set), len(done_sra_set), len(undone_sra_set), len(not_exist_sra_set)))

print("-"*25 + "Parsing Lat Lon Done" + "-"*25)