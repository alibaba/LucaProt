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
import json
import os
from bs4 import BeautifulSoup


def create_doc_from_filename(filename):
    # create BeautifulSoup from html
    with open(filename, "r", encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
    return soup


def parse(soup):
    obj = {}
    title_div = soup.find_all("h2", class_="title")
    for div in title_div:
        title = div.text
        obj["title"] = title
    doc_sum = soup.find("div", class_="docsum")
    for dl in doc_sum.findChildren("dl"):
        dt = dl.find("dt").text.strip()
        dd = dl.find("dd")
        tr_list = dd.find_all("tr")
        if tr_list:
            obj[dt] = {}
            for tr in tr_list:
                th = tr.find("th").text.strip()
                td = tr.find("td").text.strip()
                obj[dt][th] = td
        else:
            obj[dt] = dd.text.strip()
    return obj


if __name__ == "__main__":
    with open("./data/00all_SRA_run_biosample_info_res.txt", "w") as rfp:
        for filename in os.listdir("./sra_biosample_html"):
            if filename.endswith(".html"):
                sra_biosample = filename.replace(".html", "").split("_")
                sra = sra_biosample[0]
                biosample = sra_biosample[1]
                soup = create_doc_from_filename(os.path.join("./sra_biosample_html/", filename))
                try:
                    obj = parse(soup)
                except Exception as e:
                    print(e)
                    print("fail to parse html: %s" % filename)
                    continue
                obj["sra"] = sra
                obj["biosample"] = biosample
                if "Attributes" in obj:
                    rfp.write(json.dumps(obj, ensure_ascii=False)+"\n")
                else:
                    print(filename)
                    print(obj)
    print("-"*25 + "Extract Attr Done" + "-"*25)