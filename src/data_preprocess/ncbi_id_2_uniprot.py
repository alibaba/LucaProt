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
@datetime: 2022/12/9 17:39
@project: DeepProtFunc
@file: ncbi_id_2_uniprot
@desc: xxxx
'''
import csv
import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry

'''
reference：https://www.uniprot.org/help/id_mapping
'''


POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"


retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise


def submit_id_mapping(from_db, to_db, ids):
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    check_response(request)
    return request.json()["jobId"]


def get_next_link(headers):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def check_id_mapping_results_ready(job_id):
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        check_response(request)
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] == "RUNNING":
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(j["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def get_batch(batch_response, file_format, compressed):
    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)


def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    check_response(request)
    return request.json()["redirectURL"]


def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text


def get_xml_namespace(element):
    m = re.match(r"\{(.*)\}", element.tag)
    return m.groups()[0] if m else ""


def merge_xml_results(xml_results):
    merged_root = ElementTree.fromstring(xml_results[0])
    for result in xml_results[1:]:
        root = ElementTree.fromstring(result)
        for child in root.findall("{http://uniprot.org/uniprot}entry"):
            merged_root.insert(-1, child)
    ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
    return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)


def print_progress_batches(batch_index, size, total):
    n_fetched = min((batch_index + 1) * size, total)
    print(f"Fetched: {n_fetched} / {total}")


def get_id_mapping_results_search(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    if "size" in query:
        size = int(query["size"][0])
    else:
        size = 500
        query["size"] = size
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))
    url = parsed.geturl()
    request = session.get(url)
    check_response(request)
    results = decode_results(request, file_format, compressed)
    total = int(request.headers["x-total-results"])
    print_progress_batches(0, size, total)
    for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
        results = combine_batches(results, batch, file_format)
        print_progress_batches(i, size, total)
    if file_format == "xml":
        return merge_xml_results(results)
    return results


def get_id_mapping_results_stream(url):
    if "/stream/" not in url:
        url = url.replace("/results/", "/results/stream/")
    request = session.get(url)
    check_response(request)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)


def extract_id_for_rdrp(s, type="rdrp"):
    '''
    extract id for protein sequence name
    :param s: name
    :param type: protein type
    :return:
    '''
    if type == "rdrp":
        '''
        begin_idx = s.find("like_")
        if begin_idx < 0:
            begin_idx = s.find("_")
            if begin_idx >= 0:
                begin_idx += 1
        else:
            begin_idx += 5
        print("begin_idx:", begin_idx)
        if begin_idx < 0:
            return None
        end_idx = s[begin_idx:].find(".")
        print("end_idx:", end_idx)
        if end_idx < 1:
            return None
        return s[begin_idx: end_idx+begin_idx]
        '''
        end_idx = len(s)
        for idx in range(len(s) - 1, -1, -1):
            if '0' <= s[idx] <= '9' and idx > 0 and s[idx - 1] == '.':
                end_idx = idx - 1
                break
        begin_idx = 0
        has_alpha = False
        for idx in range(end_idx - 2, -1, -1):
            if s[idx] == "_" and has_alpha:
                begin_idx = idx + 1
                break
            elif 'a' <= s[idx] <= 'z' or 'A' <= s[idx] <= 'Z':
                has_alpha = True
        return s[begin_idx:end_idx]
    elif type == "other_virus" or type == "non_virus":
        strs = s.split("|")
        if len(strs) < 2:
            return None
        return strs[1]
    return None


def get_from_api(ids, data, update_idx):
    job_id = submit_id_mapping(
        from_db="RefSeq_Protein", to_db="UniProtKB", ids=ids
    )
    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        results = get_id_mapping_results_search(link)
        # Equivalently using the stream endpoint which is more demanding
        # on the API and so is less stable:
        # results = get_id_mapping_results_stream(link)
        # {'results': [{'from': 'P05067', 'to': 'CHEMBL2487'}], 'failedIds': ['P12345']}
        if results:
            for item in results["results"]:
                from_id = item["from"]
                to = item["to"]
                if isinstance(to, dict):
                    to_id = to["primaryAccession"]
                else:
                    to_id = to
                data[from_id][update_idx] = to_id
    return data


if __name__ == "__main__":
    rdrp_fasta_list = [
        "../data/rdrp/RdRp20211115.fasta",
        "../data/rdrp/other_virus_pro_sequence.fasta",
        "../data/rdrp/non_virus_sequence.fasta"
    ]
    rdrp_ids_list = [
        "../data/rdrp/RdRp20211115_id.csv",
        "../data/rdrp/other_virus_id.csv",
        "../data/rdrp/non_virus_id.csv"
    ]
    for idx, rdrp_fasta_filepath in enumerate(rdrp_fasta_list):
        with open(rdrp_ids_list[idx], "w") as wfp:
            writer = csv.writer(wfp)
            writer.writerow(["ori_id", "RefSeq_Protein", "UniProtKB"])
            with open(rdrp_fasta_filepath, "r") as rfp:
                for line in rfp:
                    line = line.strip()
                    if line.startswith(">"):
                        uuid = line.strip()
                        if idx == 0:
                            refseq_id = extract_id_for_rdrp(uuid, type="rdrp")
                            writer.writerow([uuid, refseq_id, None])
                        elif idx == 1:
                            uniprot_id = extract_id_for_rdrp(uuid, type="other_virus")
                            writer.writerow([uuid, None, uniprot_id])
                        elif idx == 2:
                            uniprot_id = extract_id_for_rdrp(uuid, type="non_virus")
                            writer.writerow([uuid, None, uniprot_id])

    data = {}
    all_ids = []
    with open("../data/rdrp/RdRp20211115_id.csv", "r") as rfp:
        reader = csv.reader(rfp)
        cnt = 0
        from_db = "RefSeq_Protein"
        to_db = "UniProtKB"
        ids = []
        for row in reader:
            cnt += 1
            if cnt == 1:
                continue
            ids.append(row[1])
            data[row[1]] = row
            if len(ids) == 1000:
                data = get_from_api(ids, data, update_idx=-1)
                all_ids.extend(ids)
                ids = []
        if len(ids) > 0:
            all_ids.extend(ids)
            data = get_from_api(ids, data, update_idx=-1)

    total = 0
    with open("../data/rdrp/RdRp20211115_id.csv", "w") as wfp:
        writer = csv.writer(wfp)
        writer.writerow(["ori_id", "RefSeq_Protein", "UniProtKB"])
        for id in all_ids:
            writer.writerow(data[id])
            if data[id][-1] is None or len(data[id][-1]) == 0:
                total += 1
        print("unfound num: %d" %total)


    '''
    job_id = submit_id_mapping(
        to_db="RefSeq_Protein", from_db="UniProtKB_AC-ID", ids=["B6QWN5", "A0A182G8Y9"]
    )
    job_id = submit_id_mapping(
        to_db="RefSeq_Protein", from_db="UniProtKB-Swiss-Prot", ids=["A0A182G8Y9"]
    )
    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        results = get_id_mapping_results_search(link)
        # Equivalently using the stream endpoint which is more demanding
        # on the API and so is less stable:
        # results = get_id_mapping_results_stream(link)
    
    print(results)
    # {'results': [{'from': 'P05067', 'to': 'CHEMBL2487'}], 'failedIds': ['P12345']}
    '''
    job_id = submit_id_mapping(
        from_db="RefSeq_Protein", to_db="UniProtKB", ids=["NP_056758", "NP_044727", "B6QWN5"]
    )
    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        results = get_id_mapping_results_search(link)
        # Equivalently using the stream endpoint which is more demanding
        # on the API and so is less stable:
        # results = get_id_mapping_results_stream(link)
        print(results)
