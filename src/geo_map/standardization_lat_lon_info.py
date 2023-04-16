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
@datetime: 2023/3/14 18:55
@project: DeepProtFunc
@file: standardization_lat_lon_info
@desc: standardization lat and lon
'''

import re, csv
import math
import pandas as pd

lat_lon_p = r"[.\d]+ *[NS] *[\d.]+ *[EW]"


def is_none(s):
    '''
    judge that the input is null
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


def transform(s):
    # print("transform:", s)
    strs = []
    cur = ""
    s = s.strip()
    for ch in s:
        if '0' <= ch <= '9' or ch in ['.', '-']:
            cur += ch
        else:
            if cur:
                strs.append(cur)
            cur = ""
    if cur:
        strs.append(cur)
        # print(strs)
    if len(strs) > 1:
        value = float(strs[0])
        for idx in range(1, len(strs)):
            value += float(strs[idx])/math.pow(60, idx)
    elif len(strs) == 1:
        value = t_float(strs[0])
    else:
        value = None
    # print(s, ":", strs, ":", value)
    return value


def t_float(s):
    point_num = 0
    last_idx = -1
    for idx, ch in enumerate(s):
        if ch == ".":
            point_num += 1
            last_idx = idx
    if point_num >= 2:
        idx = s[:last_idx].rfind(".")

        if s[0] == "-":
            print("two: ", s, "-", -float(s[idx+1:]))
            return -float(s[idx+1:])
        else:
            print("two: ", s, float(s[idx+1:]))
            return float(s[idx+1:])
    return float(s)


def extract_lat_lon(s):
    if not is_none(s):
        new_s = re.search(lat_lon_p, s)
        if new_s:
            new_s = new_s.group()
            # print("re:", s, s.replace(new_s, ""), new_s)
            return s.replace(new_s, "").strip(), new_s
    return s, None


def lat_lon_split(s):
    if is_none(s):
        return None, None
    if "* " in s or "&amp;#176" in s:
        s = s.replace("' ", "'")
    s = s.replace("degrees", "度").replace("* ", "度").replace("&amp;#176; ", "度")
    s = s.upper().strip()
    if "N" not in s and "S" not in s and "E" not in s and "W" not in s:
        strs = re.split("[ /]", s)
        strs = [v.strip() for v in strs if not is_none(v.strip())]
        if len(strs) != 2:
            print(s)
        lat, lon = transform(strs[0]), transform(strs[1])
        if is_none(lat) or is_none(lon):
            print(s, lat, lon)
            assert 1 == 0
    else:
        new_s = re.search(lat_lon_p, s)
        if new_s:
            # print("re:", s, new_s)
            s = new_s.group()
        lat = None
        lon = None
        cur = ""
        for ch in s:
            if ch == "N":
                lat = transform(cur)
                cur = ""
            elif ch == "S":
                lat = -transform(cur)
                cur = ""
            elif ch == "E":
                lon = transform(cur)
                cur = ""
            elif ch == "W":
                lon = -transform(cur)
                cur = ""
            else:
                cur += ch
        if is_none(lon) and cur:
            lon = transform(cur)
    # print(s,":", lat,":", lon)
    return lat, lon


def average(s1, s2, threshold):
    if (is_none(s1) or abs(float(s1)) > threshold) and (is_none(s2) or abs(float(s2)) > threshold):
        return None
    if is_none(s1) or abs(float(s1)) > threshold:
        return float(s2)
    if is_none(s2) or abs(float(s2)) > threshold:
        return float(s1)
    return (float(s1) + float(s2))/2


if __name__ == "__main__":
    all_lat_lon = set()
    geo_loc_name_set = set()
    marine_region_set = set()
    geographic_location_set = set()
    lat_lon_set = set()
    geographic_location_latitude_set = set()
    geographic_location_longitude_set = set()
    geographic_location_depth_set = set()
    latitude_start_set = set()
    latitude_end_set = set()
    longitude_start_set = set()
    longitude_end_set = set()
    sra_info_exists = set()
    sra_non_lat_lon = set()
    sra_non = set()
    sra_loc_info = {}
    non_lat_lon_name = set()
    sra_stats = [set(), set()]
    position_lat_lon_manual = {}
    with open("./data/position_lat_lon_manual.txt", "r") as rfp:
        reader = csv.reader(rfp)
        cnt = 0
        for row in reader:
            cnt += 1
            if cnt == 1:
                continue
            name, lon, lat = row[0], row[1], row[2]
            position_lat_lon_manual[name] = [float(lat), float(lon)]
    print("position_lat_lon_manual: %d" % len(position_lat_lon_manual))
    sra_lat_lon_manual = {}
    with open("./data/sra_lat_lon_manual.txt", "r") as rfp:
        cnt = 0
        for line in rfp:
            cnt += 1
            if cnt % 2 == 1:
                sra = line.split(" ")[0].strip()
            else:
                strs = line.strip().split(",")
                name, lon_lat = strs[0], strs[1]
                sra_lat_lon_manual[sra] = lon_lat
    print("sra_lat_lon_manual: %d" %len(sra_lat_lon_manual))

    filename = "./data/00all_SRA_run_res_simple_all.csv"
    with open(filename, "r") as rfp:
        reader = csv.reader(rfp)
        cnt = 0
        for row in reader:
            cnt += 1
            if cnt == 1:
                continue
            value_type = "ori"
            SRA_Run, biosample, geo_loc_name, marine_region, geographic_location, lat_lon, \
            geographic_location_latitude, geographic_location_longitude, geographic_location_depth, latitude_start, latitude_end, longitude_start, longitude_end, center_name = row
            geo_loc_name, cur_lat_lon1 = extract_lat_lon(geo_loc_name)
            geographic_location, cur_lat_lon2 = extract_lat_lon(geographic_location)
            # print(SRA_Run, biosample, geographic_location_latitude, geographic_location_longitude)
            # selection strategy of location name
            # first choice: geo_loc_name，
            # second choice: geographic_location，
            # third choice: marine_region，
            # final choice: center_name（center_name need to complete）

            # election strategy of the latitude and longitude
            # first choice: lat_lon,
            # second choice:(geographic_location_latitude， geographic_location_longitude),
            # final choice:the mean value of (latitude_start, latitude_end, longitude_start, longitude_end)
            # print(geographic_location)
            sra_info_exists.add(SRA_Run)
            geo_loc_name_set.add(geo_loc_name)
            marine_region_set.add(marine_region)
            geographic_location_set.add(geographic_location)
            lat_lon_set.add(lat_lon)
            geographic_location_latitude_set.add(geographic_location_latitude)
            geographic_location_longitude_set.add(geographic_location_longitude)
            geographic_location_depth_set.add(geographic_location_depth)
            latitude_start_set.add(latitude_start)
            latitude_end_set.add(latitude_end)
            longitude_start_set.add(longitude_start)
            longitude_end_set.add(longitude_end)
            name = None
            if not is_none(geo_loc_name):
                name = geo_loc_name
            if is_none(name) and not is_none(geographic_location):
                name = geographic_location
            if is_none(name) and not is_none(marine_region):
                name = marine_region
            '''
            if is_none(name) and not is_none(center_name):
                name = center_name
            '''
            '''
            if is_none(name):
                print(row)
            '''
            lat, lon = None, None
            if not is_none(lat_lon):
                lat, lon = lat_lon_split(lat_lon)
            if is_none(lat) or is_none(lon):
                if not is_none(cur_lat_lon1):
                    lat, lon = lat_lon_split(cur_lat_lon1)
            if is_none(lat) or is_none(lon):
                if not is_none(cur_lat_lon2):
                    lat, lon = lat_lon_split(cur_lat_lon2)

            if is_none(lat) and not is_none(geographic_location_latitude):
                lat = transform(geographic_location_latitude)
            if is_none(lat):
                lat = average(latitude_start, latitude_end, 90)

            if is_none(lon) and not is_none(geographic_location_longitude):
                lon = transform(geographic_location_longitude)
            if is_none(lon):
                lon = average(longitude_start, longitude_end, 180)
            if is_none(lat) or is_none(lon):
                if SRA_Run in sra_lat_lon_manual:
                    lat, lon = lat_lon_split(sra_lat_lon_manual[SRA_Run])
                    value_type = "manual_by_sra"
                elif not is_none(name) and name in position_lat_lon_manual:
                    lat, lon = position_lat_lon_manual[name]
                    value_type = "manual_by_loc_name"
                else:
                    sra_non_lat_lon.add(SRA_Run)
                    if is_none(name):
                        sra_non.add(SRA_Run)
                    else:
                        non_lat_lon_name.add(name)
                    continue

            if abs(lat) > 90 or abs(lon) > 180:
                print("invalid: ")
                print(row)
                continue

            sra_loc_info[SRA_Run] = [name, lat, lon, value_type]
            all_lat_lon.add("%0.8f###%0.8f" % (lat, lon))
    # 1801
    print("all_lat_lon size: %d" % len(all_lat_lon))
    # 10437（all sra htmls are ok)
    print("sra_loc_info size: %d" % len(sra_info_exists))
    # 492 non lat lon
    print(len(sra_non_lat_lon.difference(sra_loc_info.keys())))
    print(sra_non_lat_lon.difference(sra_loc_info.keys()))
    # 493 non loc non lat lon
    print("sra non size: %d" % len(sra_non))
    print(sra_non)
    # 492
    print(len(sra_non.difference(sra_loc_info.keys())))
    print(sra_non.difference(sra_loc_info.keys()))

    # 1
    # not determined
    print(non_lat_lon_name)
    print(len(non_lat_lon_name))
    for v in non_lat_lon_name:
        print(v)
    print("cnt: %d, exists info size: %d, not exists lat_lon: %d, exists lat_lon size: %d" %(cnt -1,
                                                                                             len(sra_info_exists),
                                                                                             len(sra_non_lat_lon.difference(sra_loc_info.keys())), len(sra_loc_info)))
    self_testing_run_ids = set()
    df = pd.read_excel(io='data/50lib_info.xlsx')
    for index, row in df.iterrows():
        run_id = row["RNA编号"].strip()
        name = row["采集地点"].strip()
        lat = float(row["纬度"])
        lon = float(row["经度"])
        all_lat_lon.add("%0.8f###%0.8f" %(lat, lon))
        sra_loc_info[run_id] = [name, lat, lon, "self_testing"]
    print("all_lat_lon size: %d" % len(all_lat_lon))

    all_lat_lon2 = set()
    with open("./data/all_sra_lat_lon.csv", "w") as wfp:
        writer = csv.writer(wfp)
        writer.writerow(["sra", "name", "lat", "lon", "type"])
        for item in sra_loc_info.items():
            if abs(item[1][1]) > 90 or abs(item[1][2]) > 180:
                print("invalid: ")
                print(row)
            writer.writerow([item[0]] + item[1])
            v = "%0.8f###%0.8f" % (item[1][1], item[1][2])
            all_lat_lon2.add(v)
    print("all_lat_lon size: %d" % len(all_lat_lon2))
    print(all_lat_lon.difference(all_lat_lon2))

    print("-"*25 + "Standardization Lat Lon Done" + "-"*25)
