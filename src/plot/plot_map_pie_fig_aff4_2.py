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
@file: plot_map_pie_fig_aff4_2
@desc: plot map pie
'''
import os, csv
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from  math import radians
from math import tan,atan,acos,sin,cos,asin,sqrt
from scipy.spatial.distance import pdist, squareform
sns.set()

dir_path = "/mnt_nas/****/rdrp/predicts/rdrp_40_extend/protein/binary_class/sefn/20230201140320-bak/checkpoint-100000/all/"

# The species corresponding to each virus
# The virus_id below removes the semicolon and is the contig_id
contig_id_2_species_id = {}
contig_id_set = set()
species_id_set = set()
with open(os.path.join(dir_path, "ecology/all_species.txt.edit"), "r") as rfp:
    cnt = 0
    reader = csv.reader(rfp, delimiter='\t')
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        contig_id, species_id = row[0].strip(), row[1].strip()
        contig_id_set.add(contig_id)
        species_id_set.add(species_id)
        if contig_id not in contig_id_2_species_id:
            contig_id_2_species_id[contig_id] = set()
        contig_id_2_species_id[contig_id].add(species_id)
print("contig size: %d" %len(contig_id_set))
print("species size: %d" %len(species_id_set))

# Check whether the contig crosses species, result: no cross
for item in contig_id_2_species_id.items():
    assert len(item[1]) == 1

# Verify that all virus_ids exist in species
with open(os.path.join(dir_path, "ecology/all_lib_rpm_cov.txt.cov20"), "r") as rfp:
    reader = csv.reader(rfp, delimiter="\t")
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        superclade_name, virus_id, rpm, coverage, lib = row[0], row[1], row[2], row[3], row[4]
        virus_id = virus_id[0:virus_id.index(':')]
        assert virus_id in contig_id_set

# Get the group where each superclade is located
df = pd.read_csv(os.path.join(dir_path, "tbl/all_info.tbl"), delimiter='\t')
superclade_group = {}
group_set = set()
for index, row in df.iterrows():
    superclade, cluster, protein_id, group1, group2 = row["Superclade"], row["Cluster"], row["ID"], row["group1"], row["group2"]
    if superclade not in superclade_group:
        superclade_group[superclade] = set()
    superclade_group[superclade].add(group2)
    group_set.add(group2)
print("superclade size: %d" %len(superclade_group))
print("group size: %d" %len(group_set))
print(group_set)

# Check whether the superclade crosses the group, the result: no cross
for item in superclade_group.items():
    assert len(item[1]) == 1

# Get the latitude and longitude of each sra
filepath = "/mnt_nas/****/workspace/DeepProtFunc/src/geo/data/all_sra_lat_lon.csv"
sra_lat_lon_info = {}
lat_lon_sra = {}
sra_set = set()
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


# The SRA set with lat and lon
found_set = set()
# The SRA set with no lat and lon
unfound_set = set()
# The latitude and longitude involved in each species
species_lat_lon = {}
# Species under each group (label) under each latitude and longitude
lat_lon_species = {}
# species at each latitude and longitude
lat_lon_species = {}
lat_lon_species_no_label = {}
all_contigs = set()
all_species = set()
all_virus = set()
with open(os.path.join(dir_path, "ecology/all_lib_rpm_cov.txt.cov20.res_v5.csv"), "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["superclade_name", "virus_id", "rpm", "coverage", "lib", "contig_id", "species_id", "location_name", "lat", "lon", "lat_lon_type"])
    with open(os.path.join(dir_path, "ecology/all_lib_rpm_cov.txt.cov20"), "r") as rfp:
        reader = csv.reader(rfp, delimiter="\t")
        cnt = 0
        for row in reader:
            cnt += 1
            if cnt == 1:
                continue
            superclade_name, virus_id, rpm, coverage, lib = row[0], row[1], row[2], row[3], row[4]
            contig_id = virus_id[:virus_id.index(":")]
            species_id = list(contig_id_2_species_id[contig_id])[0]
            # Divide into two groups according to the value of group value
            group = list(superclade_group[superclade_name])[0]
            if group == "known_blast_hmm_ai":
                label = "know"
            elif group in ["novel_blast_hmm_ai", "novel_hmm_ai"]:
                label = "cluster"
            else:
                label = "ai_specific"
            if lib in sra_lat_lon_info:
                writer.writerow([superclade_name, virus_id, rpm, coverage, lib, contig_id, species_id] + sra_lat_lon_info[lib])
                lat_lon = "%0.8f###%0.8f" %(sra_lat_lon_info[lib][0], sra_lat_lon_info[lib][1])
                if species_id not in species_lat_lon:
                    species_lat_lon[species_id] = set()
                species_lat_lon[species_id].add(lat_lon)

                if lat_lon not in lat_lon_species:
                    lat_lon_species[lat_lon] = {"know": set(), "cluster": set(), "ai_specific": set()}
                    lat_lon_species_no_label[lat_lon] = set()
                lat_lon_species[lat_lon][label].add(species_id)
                lat_lon_species_no_label[lat_lon].add(species_id)

                all_contigs.add(contig_id)
                all_species.add(species_id)
                all_virus.add(virus_id)
                found_set.add(lib)
            else:
                unfound_set.add(lib)
print("total sra size: %d, found size: %d, unfound size: %d" %(len(found_set.union(unfound_set)), len(found_set), len(unfound_set)))
print("contigs size: %d, species size: %d, virus size: %d" %(len(all_contigs), len(all_species), len(all_virus)))


data = {}
groups = ["know", "cluster", "ai_specific"]
species_set = set()
# The number of species under each latitude and longitude
for item in lat_lon_species.items():
    lat_lon = item[0]
    if lat_lon not in data:
        data[lat_lon] = {}
        for label in groups:
            data[lat_lon][label] = len(item[1][label])
            species_set = species_set.union(item[1][label])
print("species size: %d" % len(species_set))


with open("map/data_v5.csv", "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["lat", "lon", "know_species_num", "cluster_species_num", "ai_specific_species_num", "species_num"])
    for item in data.items():
        lat_lon = item[0]
        strs = lat_lon.split("###")
        lat = float(strs[0])
        lon = float(strs[1])
        if abs(lat) > 90:
            raise Exception(lat, lon)
        if abs(lon) > 180:
            raise Exception(lat, lon)
        know_species_num = item[1]["know"]
        cluster_species_num = item[1]["cluster"]
        ai_specific_species_num = item[1]["ai_specific"]
        assert len(lat_lon_species_no_label[lat_lon]) == int(know_species_num + cluster_species_num + ai_specific_species_num)
        writer.writerow([lat, lon, know_species_num, cluster_species_num, ai_specific_species_num, know_species_num + cluster_species_num + ai_specific_species_num])


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


# load data
data = []
with open("map/data_v5.csv", "r") as rfp:
    reader = csv.reader(rfp)
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        # print(row)
        lat, lon, know_species_num, cluster_species_num, ai_specific_species_num, species_num = row[0], row[1], row[2], row[3], row[4], row[5]
        data.append([lat, lon, know_species_num, cluster_species_num, ai_specific_species_num, species_num])

print("data size: %d" %len(data))
data = np.array(data)
df = pd.DataFrame(data, columns=["lat", "lon", "know_species_num", "cluster_species_num", "ai_specific_species_num", "species_num"], dtype=float)

distance_matrix = squareform(pdist(df, (lambda row1, row2: haversine((row1[0], row1[1]), (row2[0], row2[1])))))

# distance threshold（km）
distance_threshold = 800
db_res = DBSCAN(eps=distance_threshold,
                min_samples=1,
                metric='precomputed').fit_predict(distance_matrix)

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
df.to_csv("map/data_dbscan_v5.csv", encoding='utf-8')

group_df = df.groupby('label')[['lat', 'lon']].mean()

# Save the cluster id to which each point belongs
group_df.to_csv("map/data_dbscan_group_latlng_v5.csv", encoding='utf-8')
group_df.reset_index(inplace=True)

sum_df = df.groupby('label')["species_num"].sum()

sum_df.to_csv("map/data_dbscan_group_sum_v5.csv", encoding='utf-8')

df = pd.read_csv("map/data_dbscan_v5.csv", delimiter=',')

cluster_data = {}

for index, row in df.iterrows():
    label = int(row["label"])
    lat = row["lat"]
    lon = row["lon"]
    know_species_num = row["know_species_num"]
    cluster_species_num = row["cluster_species_num"]
    ai_specific_species_num = row["ai_specific_species_num"]
    if label not in cluster_data:
        cluster_data[label] = []
    cluster_data[label].append([lat, lon, know_species_num, cluster_species_num, ai_specific_species_num])


new_data = []
species_num_list = []
with open("map/cluster_data_v2.csv", "w") as wfp:
    writer = csv.writer(wfp)
    writer.writerow(["label", "lat", "lon", "know_species_num", "cluster_species_num", "ai_specific_species_num", "species_num"])
    for item1 in cluster_data.items():
        label = item1[0]
        avg_lat = 0
        avg_lon = 0
        species_num = 0
        know_species_num = 0
        cluster_species_num = 0
        ai_specific_species_num = 0
        for item2 in item1[1]:
            species_num += item2[2]
            species_num += item2[3]
            species_num += item2[4]
            know_species_num += item2[2]
            cluster_species_num += item2[3]
            ai_specific_species_num += item2[4]
            avg_lat += item2[0]
            avg_lon += item2[1]
        avg_lat /= len(item1[1])
        avg_lon /= len(item1[1])
        assert int(species_num) == int(know_species_num + cluster_species_num + ai_specific_species_num)
        new_data.append([label, avg_lat, avg_lon, know_species_num,  cluster_species_num, ai_specific_species_num, species_num])
        writer.writerow([label, avg_lat, avg_lon, know_species_num,  cluster_species_num, ai_specific_species_num, species_num])
        species_num_list.append(species_num)
print("species_num_list, sum: %d, max: %d, min: %d" %(sum(species_num_list), max(species_num_list), min(species_num_list)))
print(sorted([int(v) for v in species_num_list]))


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
proj = ccrs.PlateCarree()
# create figure
fig = plt.figure(figsize=(20, 20), dpi=600)
# create sub-figure
ax = plt.axes(projection=proj)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE, lw=0.5)
ax.add_feature(cfeature.RIVERS, lw=0.5)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.BORDERS, lw=0.1)
for item in new_data:
    lat = item[1]
    lon = item[2]
    know_species_num = int(item[3])
    cluster_species_num = int(item[4])
    ai_specific_species_num = int(item[5])
    species_num = int(item[6])
    assert species_num == know_species_num + cluster_species_num + ai_specific_species_num
    ax_sub = inset_axes(ax, width=0.5, height=0.5, loc=10, bbox_to_anchor=(lon, lat), bbox_transform=ax.transData)
    '''
    [
        1, 1, 2, 3, 3, 3, 6, 6, 6, 7, 8, 8, 9, 
        10, 11, 12, 14, 20, 22, 22, 24, 32, 38, 41, 44, 46, 49, 52, 60, 65, 71, 78, 98, 
        101, 102, 116, 122, 139, 142, 157, 157, 169, 173, 177, 181, 184, 194, 204, 206, 217, 219, 309, 324, 409, 538, 548, 553, 672, 716, 911, 966, 996, 
        1008, 1778, 1862, 2718, 6689, 8886, 
        24634, 89619, 14469
    ]
    '''
    # < 10 <100 <1000 < 10000 10000+
    if species_num < 10:
        radius = 1
    elif species_num < 100:
        radius = 2
    elif species_num < 1000:
        radius = 3
    elif species_num < 10000:
        radius = 4
    else:
        radius = 5

    radius /= 2
    nums = [know_species_num, cluster_species_num, ai_specific_species_num]
    colors = ["#D1D1D1", "#3BCCA2", "#FF9E2E"]
    cur_colors = []
    cur_nums = []
    for idx, v in enumerate(nums):
        if v <= 0:
            continue
        cur_nums.append(v)
        cur_colors.append(colors[idx])
    ax_sub.pie(cur_nums,
               wedgeprops={'edgecolor': '#333333', 'linewidth': 0.25, 'antialiased': True},
               radius=radius,
               colors=colors)

plt.savefig('map_pie_rdrp_species_3groups_v5_no_trans_level_5.pdf', dpi=600, format="pdf")
plt.savefig('map_pie_rdrp_species_3groups_v5_no_trans_level_5.png', dpi=600, format="png")
plt.show()
