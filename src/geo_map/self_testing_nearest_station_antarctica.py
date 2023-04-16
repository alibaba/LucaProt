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
@datetime: 2023/3/15 09:55
@project: DeepProtFunc
@file: self_testing_nearest_station_antarctica
@desc: get Antarctica station of our self_testing dataset in Antarctica
'''
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
from math import sin, cos, asin, sqrt, radians
try:
    from standardization_lat_lon_info import *
except ImportError:
    from src.geo_map.standardization_lat_lon_info import *


def haversine(latlon1, latlon2):
    '''
    calc the distance by latitude and longitude
    '''
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2
    lon1, lat1, lon2, lat2 = radians(lon1), radians(lat1), radians(lon2), radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles

    dist = c * r # # km
    return dist


df = pd.read_excel(io='data/antarctic_research_station.xlsx')
antarctic_station = {}
for index, row in df.iterrows():
    name = row["名称"]
    lat_lon = row["坐标"]
    nation = str(row["国家"]).strip()
    print(nation)
    if is_none(lat_lon):
        continue
    if nation != "中国":
        continue
    antarctic_station[name] = lat_lon_split(lat_lon)
    print(name, lat_lon, antarctic_station[name])

df = pd.read_excel(io='data/50lib_info.xlsx')
want_calc = []
for index, row in df.iterrows():
    sra = row["RNA编号"]
    lat = float(row["纬度"])
    lon = float(row["经度"])
    loc = row["采集地点"].strip()
    if loc == "南极":
        dist_list = []
        for item in antarctic_station.items():
            dist = haversine((lat, lon), (item[1][0], item[1][1]))
            dist_list.append([item[0], item[1], dist])
        dist_list = sorted(dist_list, key=lambda x: x[-1])
        print(sra, lat, lon)
        print(dist_list)


