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
@datetime: 2023/3/14 14:20
@project: DeepProtFunc
@file: plot_map_pie_fig4_1
@desc: plot map pie
'''
import matplotlib.pyplot as plt
import os, csv
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn import metrics
from math import sin, cos, asin, sqrt, radians
from scipy.spatial.distance import pdist, squareform
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set()


dir_path = "/mnt_nas/****/rdrp/predicts/rdrp_40_extend/protein/binary_class/sefn/20230201140320-bak/checkpoint-100000/all/"


# Habitat for all SRAs
sra_habitat = {}
habitat_set = set()
with open(os.path.join(dir_path, "ecology/00all_sample.info.edit"), "r") as rfp:
    cnt = 0
    reader = csv.reader(rfp, delimiter='\t')
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        sra, habitat = row[1].strip(), row[2].strip()
        sra = sra.replace("2202SPSS-", "2201SPSS-")
        if sra not in sra_habitat:
            sra_habitat[sra] = set()
        sra_habitat[sra].add(habitat)
        habitat_set.add(habitat)
print("habitat size: %d" % len(habitat_set))
print("habitat: ", habitat_set)
# Check if a sra involves multiple habitats
for item in sra_habitat.items():
    assert len(item[1]) == 1


# get the lat-lng of all SRAs
filepath = "/mnt_nas/****/workspace/DeepProtFunc/src/geo/data/all_sra_lat_lon.csv"
sra_lat_lon_info = {}
lat_lon_sra = {}
sra_set = set()
self_testing_lat_lon = set()
with open(filepath, "r") as rfp:
    cnt = 0
    reader = csv.reader(rfp)
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        sra, name, lat, lon, data_type = row[0], row[1], float(row[2]), float(row[3]), row[4]
        sra_lat_lon_info[sra] = [lat, lon]
        lat_lon = "%0.8f###%0.8f" %(lat, lon)
        if lat_lon not in lat_lon_sra:
            lat_lon_sra[lat_lon] = set()
        lat_lon_sra[lat_lon].add(sra)
        sra_set.add(sra)
print("sra size: %d, lat_lon_sra size: %d" % (len(sra_set), len(lat_lon_sra)))


# Species list under each group (label) of each lib
lib_habitat = {}
# The latitude and longitude involved in each lib
lib_lat_lon = {}
# libs under each group (label) under each latitude and longitude
lat_lon_libs = {}

filepath = "/mnt_nas/****/workspace/DeepProtFunc/src/geo/data/all_sra_lat_lon.csv"
with open(filepath, "r") as rfp:
    reader = csv.reader(rfp)
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        lib, name, lat, lon, data_type = row[0], row[1], float(row[2]), float(row[3]), row[4]
        # Get the habitat corresponding to SRA
        habitat = sra_habitat[lib]
        lat_lon = "%0.8f###%0.8f" %(sra_lat_lon_info[lib][0], sra_lat_lon_info[lib][1])
        if lib not in lib_lat_lon:
            lib_lat_lon[lib] = set()
        lib_lat_lon[lib].add(lat_lon)

        if lat_lon not in lat_lon_libs:
            lat_lon_libs[lat_lon] = set()
        lat_lon_libs[lat_lon].add(lib)

data_v6 = {}
data_v6_by_habitat = {}
groups = ['Aquatic', 'Soil', 'Extreme',  'Host_microbe', 'Host_plant', 'Food',  'Engineered',   'Other']
colors = ["#F47F60", "#FFB74D", "#D3D656", "#26A69A", "#42A5F5", "#6D7EC9", "#9575CD", "#DD6895"]
# The latitude and longitude under each lib
for item in lib_lat_lon.items():
    lib = item[0]
    lat_lon = list(item[1])[0]
    cur_habitat = list(sra_habitat[lib])[0]
    # judge whether an sra exceeds a latitude and longitude
    if len(item[1]) > 1:
        raise Exception(item)

    if lat_lon not in data_v6:
        data_v6[lat_lon] = set()
    data_v6[lat_lon].add(lib)

    if lat_lon not in data_v6_by_habitat:
        data_v6_by_habitat[lat_lon] = {}
        data_v6_by_habitat[lat_lon][cur_habitat] = set()
    elif cur_habitat not in data_v6_by_habitat[lat_lon]:
        data_v6_by_habitat[lat_lon][cur_habitat] = set()
    data_v6_by_habitat[lat_lon][cur_habitat].add(lib)

libs = set()
with open("map/data_v6_by_habitat.csv", "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["lat", "lon", "habitat", "lib_num"])
    for item1 in data_v6_by_habitat.items():
        strs = item1[0].split("###")
        lat = float(strs[0])
        lon = float(strs[1])
        if abs(lat) > 90:
            raise Exception(lat, lon)
        if abs(lon) > 180:
            raise Exception(lat, lon)
        for item2 in item1[1].items():
            cur_habitat = item2[0]
            cur_libs = item2[1]
            libs = libs.union(cur_libs)
            writer.writerow([lat, lon, cur_habitat, len(cur_libs)])


assert len(libs) == len(sra_set)

with open("map/data_v6.csv", "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["lat", "lon", "lib_num"])
    for item in data_v6.items():
        strs = item[0].split("###")
        lat = float(strs[0])
        lon = float(strs[1])
        if abs(lat) > 90:
            raise Exception(lat, lon)
        if abs(lon) > 180:
            raise Exception(lat, lon)
        writer.writerow([lat, lon, len(item[1])])


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

# print(haversine((transform("39°56′"), transform("116°20′")), (transform("23.13′"), transform("113.27"))))
## load the data
data_v6 = []
with open("map/data_v6.csv", "r") as rfp:
    reader = csv.reader(rfp)
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        # print(row)
        lat, lon, lib_num = row[0], row[1], row[2]
        data_v6.append([lat, lon, lib_num])

print("data size: %d" %len(data_v6))
data_v6 = np.array(data_v6)
df = pd.DataFrame(data_v6, columns=["lat", "lon", "lib_num"], dtype=float)
print(df)

distance_matrix = squareform(pdist(df, (lambda row1, row2: haversine((row1[0], row1[1]), (row2[0], row2[1])))))

# distance threshold（km）
distance_threshold = 800
db_res = DBSCAN(eps=distance_threshold, min_samples=1, metric='precomputed').fit_predict(distance_matrix)

labels = db_res
raito = len(labels[labels[:] == -1]) / len(labels)  # Calculate the ratio of the number of noise points to the total
print("dbscan noise: %f" % raito)
if -1 in labels:
    print("exists outlier")
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # Get the number of clusters
score = metrics.silhouette_score(distance_matrix, labels)
print("dbscan score: %f" % score)
df["label"] = labels
print("data size: ", len(labels), "clusters: ", n_clusters_)

# Save the cluster id to which each point belongs
df.to_csv("map/data_dbscan_v6.csv", encoding='utf-8')

assert int(df["lib_num"].sum()) == len(sra_set)

group_df = df.groupby('label')[['lat', 'lon']].mean()

# Save the cluster id to which each point belongs
group_df.to_csv("map/data_dbscan_group_latlng_v6.csv", encoding='utf-8')
group_df.reset_index(inplace=True)

sum_df = df.groupby(['label'])["lib_num"].sum()

sum_df.to_csv("map/data_dbscan_group_sum_v6.csv", encoding='utf-8')

df = pd.read_csv("map/data_dbscan_v6.csv", delimiter=',')

cluster_data_v6 = {}

for index, row in df.iterrows():
    label = int(row["label"])
    lat = row["lat"]
    lon = row["lon"]
    lib_num = int(row["lib_num"])
    lat_lon = "%0.8f###%0.8f" %(lat, lon)
    habitat_stats = [0] * len(groups)
    for idx, cur_habitat in enumerate(groups):
        if cur_habitat in data_v6_by_habitat[lat_lon]:
            habitat_stats[idx] = len(data_v6_by_habitat[lat_lon][cur_habitat])
    assert int(sum(habitat_stats)) == int(lib_num)
    if label not in cluster_data_v6:
        cluster_data_v6[label] = []
    cluster_data_v6[label].append([lat, lon, habitat_stats, lib_num])

species_num_list = []
new_data = []
with open("map/cluster_data_v6.csv", "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["label", "lat", "lon", "habitat_stats", "lib_num"])
    for item1 in cluster_data_v6.items():
        label = item1[0]
        avg_lat = 0
        avg_lon = 0
        lib_num = 0
        habitat_stats = [0] * len(groups)
        for item2 in item1[1]:
            avg_lat += item2[0]
            avg_lon += item2[1]
            cur_habitat_stats = item2[2]
            for idx in range(len(groups)):
                habitat_stats[idx] += cur_habitat_stats[idx]
            lib_num += item2[3]
        avg_lat /= len(item1[1])
        avg_lon /= len(item1[1])
        new_data.append([label, avg_lat, avg_lon, habitat_stats, lib_num])
        writer.writerow([label, avg_lat, avg_lon, habitat_stats, lib_num])
        species_num_list.append(lib_num)
print("species_num_list, sum: %d, max: %d, min: %d" %(sum(species_num_list), max(species_num_list), min(species_num_list)))
print(sorted([int(v) for v in species_num_list]))

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
proj = ccrs.PlateCarree()
# create figure
fig = plt.figure(figsize=(20, 20), dpi=600)
# create sub-figure
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE, lw=0.5)
ax.add_feature(cfeature.RIVERS, lw=0.5)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.BORDERS, lw=0.1)

for item in new_data:
    lat = item[1]
    lon = item[2]
    habitat_stats = item[3]
    lib_num = item[4]
    assert sum(habitat_stats) == lib_num
    ax_sub = inset_axes(ax, width=0.5, height=0.5, loc=10, bbox_to_anchor=(lon, lat), bbox_transform=ax.transData)
    '''
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7, 8, 8, 9, 9, 
    10, 11, 11, 14, 17, 18, 20, 21, 24, 25, 26, 28, 29, 30, 31, 31, 33, 35, 35, 36, 37, 52, 61, 75, 77, 89, 93, 96, 96, 97, 
    110, 129, 136, 144, 180, 285, 411, 
    
    1819, 3838
    '''
    # < 10 < 100  < 500 < 1000 1000+
    if lib_num < 10:
        radius = 1
    elif lib_num < 100:
        radius = 2
    elif lib_num < 500:
        radius = 3
    elif lib_num < 1000:
        radius = 4
    else:
        radius = 5
    radius = radius/2
    cur_habitat_stats = []
    cur_colors = []
    for idx, v in enumerate(habitat_stats):
        if v <= 0:
            continue
        cur_habitat_stats.append(v)
        cur_colors.append(colors[idx])
    ax_sub.pie(cur_habitat_stats,
               wedgeprops={'edgecolor': '#333333', 'linewidth': 0.25, 'antialiased': True},
               radius=radius,
               colors=cur_colors)

plt.savefig('map_pie_habitat_distribution_no_trans_level_5.pdf', dpi=600, format="pdf")
plt.savefig('map_pie_habitat_distribution_no_trans_level_5.png', dpi=600, format="png")

plt.show()

