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
@datetime: 2023/3/14 15:14
@project: DeepProtFunc
@file: plot_map_pie_fig_aff4_1.py
@desc: plot map pie
'''

import os, csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

dir_path = "/mnt_nas/****/rdrp/predicts/rdrp_40_extend/protein/binary_class/sefn/20230201140320-bak/checkpoint-100000/all/"
# the biotic environment of all SRAs
habitat_sra = {}
habitat_set = set()
sra_set = set()

with open(os.path.join(dir_path, "ecology/00all_sample.info.edit"), "r") as rfp:
    cnt = 0
    reader = csv.reader(rfp, delimiter='\t')
    for row in reader:
        cnt += 1
        if cnt == 1:
            continue
        sra, habitat = row[1].strip(), row[2].strip()
        sra = sra.replace("2202SPSS-", "2201SPSS-")
        if habitat not in habitat_sra:
            habitat_sra[habitat] = set()
        habitat_sra[habitat].add(sra)
        habitat_set.add(habitat)
        sra_set.add(sra)
assert len(habitat_set) == len(habitat_sra)
print("habitat size: %d" % len(habitat_set))
print("habitat: ", habitat_set)

fig = plt.figure(figsize=(12, 5), dpi=600)

bins=20
opacity=0.15
fontsize1 = 16
fontsize2 = 18
fontsize3 = 24
fontsize4 = 14
no_fontsize = 20

labels = ['Aquatic', 'Soil', 'Extreme',  'Host_microbe', 'Host_plant', 'Food',  'Engineered',   'Other']
facecolor_list = ["#F47F60", "#FFB74D", "#D3D656", "#26A69A", "#42A5F5", "#6D7EC9", "#9575CD", "#DD6895"]

x = labels
y = [len(habitat_sra[label]) for label in labels]
legend_loc = "upper left"
total = len(sra_set)

fig.patch.set_alpha(opacity)

bar = plt.bar(x=x, height=y, color=facecolor_list)

plt.ylim(0, int(max(y) * 1.2))   # Y-axis value range
plt.ylabel("Num of Samples", fontsize=fontsize4)
# plt.xlabel("Habitat", fontsize=fontsize2)
plt.title("The Sample Number Distribution on Habitats(Sample=%d,Habitats=8)" %total, fontsize=fontsize1,  y=1)
# plt.title("%s" % order_lists[idx], y=1,loc='left', fontsize=no_fontsize, color='black', fontweight="bold")
plt.xticks(fontsize=fontsize4)
plt.yticks(fontsize=fontsize4)
# plt.legend(loc=legend_loc, fontsize=fontsize4)
for rect_idx, rect in enumerate(bar):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2,
             height*1.01,
             "%d"%len(habitat_sra[labels[rect_idx]]),
             size=fontsize4, ha="center", va="bottom", rotation=0)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

fig.tight_layout()
plt.savefig('fig4_aff_sample_habitat_distri.png', dpi=600, format="png")
plt.savefig('fig4_aff_sample_habitat_distri.pdf', dpi=600, format="pdf")
plt.show()

