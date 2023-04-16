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
@datetime: 2023/1/6 17:27
@project: DeepProtFunc
@file: predict
@desc: xxxx
'''
import os
import argparse
import pickle5
import xgboost
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import *
except ImportError:
    from src.utils import *


def calc_dataset_size(filepath):
    '''
    calc the size of the dataset
    :param filepath: the dataset filepath
    :return:
    '''
    reader = file_reader(filepath, header=True, header_filter=True)
    cnt = 0
    for _ in reader:
        cnt += 1
    return cnt


def load_dataset(filepath: str, emb_dir: str, split_size=100000):
    '''
    load the dataset from file
    :param filepath:
    :param emb_dir:
    :param split_size:
    :return:
    '''
    reader = file_reader(filepath, header=True, header_filter=True)
    id_list = []
    X = []
    for row in reader:
        emb_idx, protein_id = row[0], row[1]
        emb_filepath = os.path.join(emb_dir, "%d.pt" % emb_idx)
        data = torch.load(emb_filepath)
        embedding_info = data["bos_representations"][36].numpy()
        id_list.append(protein_id)
        X.append(embedding_info)
        if len(id_list) == split_size:
            yield id_list, np.array(X)
            id_list, X = [], []
    if id_list:
        yield id_list, np.array(X)


def load_model(model_path, model_type):
    '''
    load the model from file
    :param model_path:
    :param model_type:
    :return:
    '''
    print("#" * 25 + "loading model" + "#" * 25)
    if model_type == "dnn":
        model = torch.load(model_path)
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        return model
    elif model_type in ["xgb", "lgbm"]:
        with open(model_path, "rb") as fin:
            model = pickle5.load(fin)
        return model


def predict(args, model, X):
    '''
    prediction
    :param args:
    :param model:
    :param X:
    :return:
    '''
    if args.model_type in ["xgb", "lgbm"]:
        if args.grid_search:
            return model.predict_proba(X)
        else:
            return model.predict(X)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        predict_sampler = SequentialSampler(dataset)
        predict_dataset_total_num = len(dataset)
        predict_dataloader = DataLoader(dataset, sampler=predict_sampler, batch_size=args.batch_size)
        # the predicted probs
        probs = None
        for batch in tqdm(predict_dataloader, total=predict_dataset_total_num, desc="Prediction"):
            # evaluate
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "inputs": batch[0],
                }
            outputs = model(**inputs)
            tmp_predict_loss, logits, output = outputs[:3]
            if probs is None:
                probs = output.detach().cpu().numpy()
            else:
                probs = np.append(probs, output.detach().cpu().numpy(), axis=0)
        return probs


def run(args):
    '''
    main function
    :param args:
    :return:
    '''
    # load data
    total_records_size = calc_dataset_size(args.data_path)
    dataset_num = (total_records_size + (args.split_size - 1)) // args.split_size
    dataset_iterator = trange(int(dataset_num), desc="Dataset Iterator", disable=False)
    dataset_gen = load_dataset(args.data_path, split_size=args.split_size)
    # load model
    model = load_model(args.model_path, args.model_type)
    # label labels
    label_list = load_labels(args.label_filepath, header=True)
    label_map = {idx: name for idx, name in enumerate(label_list)}
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    wfp = open(os.path.join(args.save_dir, "pred_result.csv"), "w")
    writer = csv.writer(wfp)
    writer.writerow(["protein_id", "predict_prob", "predict_label"])

    # If it is a binary classification, write a separate file for the samples predicted to be positive
    if args.task_type in ["binary-class", "binary_class"]:
        wfp_positive = open(os.path.join(args.save_dir, "pred_result_positive.csv"), "w")
        writer_positive = csv.writer(wfp_positive)
        writer_positive.writerow(["protein_id", "predict_prob", "predict_label"])
    for _ in dataset_iterator:
        cur_dataset = dataset_gen.__next__()
        id_list, X = cur_dataset
        if args.model_type == "xgb":
            X = xgboost.DMatrix(X)
        probs = predict(args, model, X)
        if args.task_type in ["multi-label", "multi_label"]:
            preds = relevant_indexes((probs >= args.threshold).astype(int))
            for idx in range(len(id_list)):
                writer.writerow([id_list[idx], probs[idx], [label_map[label_idx] for label_idx in preds[idx]]])
        elif args.task_type in ["multi-class", "multi_class"]:
            preds = np.argmax(probs, axis=1)
            for idx in range(len(id_list)):
                writer.writerow([id_list[idx], probs[idx], label_map[preds[idx]]])
        elif args.task_type == "regression":
            pass # to do
        elif args.task_type in ["binary-class", "binary_class"]:
            preds = (probs >= args.threshold).astype(int)
            for idx in range(len(id_list)):
                writer.writerow([id_list[idx], probs[idx], label_map[preds[idx]]])
                if preds[idx] == 1:
                    writer_positive.writerow([id_list[idx], probs[idx], label_map[preds[idx]]])
            wfp_positive.flush()
        else:
            raise Exception("not support task_type=%s" % args.task_type)
        wfp.flush()
    wfp.close()
    if args.task_type in ["binary-class", "binary_class"]:
        wfp_positive.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", default="binary_class", type=str, required=True, choices=["multi_label", "multi_class", "binary_class"], help="task type")
    parser.add_argument("--data_path", default=None, type=str, required=True, help="the data path")
    parser.add_argument("--split_size", default=10000, type=int, help="If there is too much data, split it according to the split size")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--model_path", default=None, type=str, required=True, help="model path")
    parser.add_argument("--model_type", default="xgb", type=str, required=True,  choices=["xgb", "lgbm", "dnn"], help="model type")
    parser.add_argument("--grid_search", action="store_true", help="whether the model to be loaded obtained through grid_search training?")
    parser.add_argument("--label_filepath", default=None, type=str, required=True, help="label filepath.")
    parser.add_argument("--save_dir", default=None, type=str, required=True, help="the path where the predicted results are saved")
    parser.add_argument("--threshold", default=None, type=str, required=True, help="For multi-label classification or binary classification, a probability threshold is required to distinguish false/true")

    args = parser.parse_args()
    args.output_mode = args.task_type
    return args


if __name__ == "__main__":
    args = main()
    run(args)
    '''
    python predict.py 
        --task_type binary_class \
        --data_path /mnt/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_001_embed_fasta_id_2_idx.csv \
        --split_size 10000 \
        --batch_size 128 \
        --model_path ../../models/rdrp_40/protein/binary_class/xgb/20230106095829/checkpoint-200/xgb_model.txt \
        --model_type xgb \
        --grid_search \
        --label_filepath ../../models/rdrp_40/protein/binary_class/xgb/20230106095829/checkpoint-200/label.txt \
        --save_dir /mnt/****/biodata/20221123-to-Ali/all_500aa.pep.split/all_500aa.part_001_xgb/ \
        --threshold 0.5
    '''