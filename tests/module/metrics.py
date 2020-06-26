#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# test evaluate metrics 
# ==================================
"""
DATA INSTANCES 
span_start: 
    e.g.: mention start indices in the original document 
    [17, 20, 26, 43, 60, 85, 86]
span_end:
    e.g.: mention end indices in the original document 
cluster_ids: 
    e.g.: cluster ids for the (span_start, span_end) pairs
    [1, 1, 2, 2, 2, 3, 3] 
check the mention in the subword list: 
    1. ['its']
    1. ['the', 'Chinese', 'securities', 'regulatory', 'department']
    2. ['this', 'stock', 'reform']
    2. ['the', 'stock', 'reform']
    2. ['the', 'stock', 'reform']
    3. ['you']
    3. ['everyone']
"""



import os 
import sys 

REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-3])
print(REPO_PATH)

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


import data_preprocess.conll as conll 
from module import metrics 
from data_loader.conll_dataloader import CoNLLDataLoader


class Config(object):
    def __init__(self, ):
        self.data_dir = "/dev/shm/xiaoya/data"
        self.sliding_window_size = 128
        self.eval_path = os.path.join(self.data_dir, "test.english.v4_gold_conll")



if __name__ == "__main__":
    config = Config()
    print(config.data_dir)
    dataloader = CoNLLDataLoader(config)

    coref_prediction_dict = {}
    subtoken_map_dict = {}
    coref_evaluator = metrics.CorefEvaluator()

    loader = dataloader.get_dataloader("test")

    for idx, case in enumerate(loader):
        cluster_collection = {}

        doc_key = loader.dataset.features[idx].doc_idx 

        tmp_doc_key = doc_key
        tmp_mention_start = case["span_start"].squeeze(0).to("cpu").numpy().tolist()
        tmp_mention_end = case["span_end"].squeeze(0).to("cpu").numpy().tolist()

        tmp_mention_span = case["mention_span"].squeeze(0).to("cpu").numpy().tolist()
        tmp_cluster_ids = case["cluster_ids"].squeeze(0).to("cpu").numpy().tolist()

        # print("=*="*10)
        # print("start test case : ")
        # print("=*="*10)

        # print("-&"*10)
        # print(idx)
        # print("doc_idx: ", tmp_doc_idx[0])
        # print("gold_mention_span: ", tmp_mention_span)
        # print("gold_cluster_ids: ", tmp_cluster_ids)


        for pointer, (start_i, end_j) in enumerate(zip(tmp_mention_start, tmp_mention_end)):
            if tmp_cluster_ids[pointer] not in cluster_collection.keys():
                cluster_collection[tmp_cluster_ids[pointer]] = [[start_i, end_j]]
            else:
                cluster_collection[tmp_cluster_ids[pointer]].append([start_i, end_j])

        cluster_key = sorted([tmp for tmp in cluster_collection.keys()])
        # print("check cluster key list: ")
        # print(cluster_key)
        cluster_pin = [cluster_collection[tmp] for tmp in cluster_key]
        # print("check cluster value list: ")
        # print(cluster_pin)

        coref_prediction_dict[tmp_doc_key] = cluster_pin
        subtoken_map_dict[tmp_doc_key] = case["subtoken_map"].squeeze(0).to("cpu").numpy().tolist()
        # coref_evaluator.update(tmp_cluster_ids, tmp_cluster_ids, tmp_mention_span, tmp_mention_span)
        # print(coref_prediction_dict)
        # print("check the prediction dict")
        # print("-*-"*10)

    summary_dict = {}
    conll_results = conll.evaluate_conll(config.eval_path, coref_prediction_dict, subtoken_map_dict, False)
    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    print("check conll evaluation results ! ")
    print(conll_results)
    print(average_f1)
    # summary_dict["Average F1 (conll)"] = average_f1 
    # print("Average F1 (conll) : {:.2f}%".format(average_f1))

    # p, r, f = coref_evaluator.get_prf()
    # summary_dict["Average F1 (py)"] = f 
    # print("Average F1 (py): {:.2f}%".format(f * 100))
    # summary_dict["Average precision (py)"] = p 
    # print("Average precision (py): {:.2f}%".format(p * 100))
    # summary_dict["Average recall (py)"] = r 
    # print("Average recall (py): {:.2f}%".format(r * 100))
    # print("###"*20)




