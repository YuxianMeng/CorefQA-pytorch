#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# test evaluate metrics 



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
    coref_evaluator = metrics.CorefEvaluator()

    loader = dataloader.get_dataloader("test")
    for idx, case in enumerate(loader):

        tmp_doc_idx = case["doc_idx"].squeeze(0).to("cpu").numpy().tolist()
        tmp_mention_span = case["mention_span"].squeeze(0).to("cpu").numpy().tolist()
        tmp_cluster_ids = case["cluster_ids"].squeeze(0).to("cpu").numpy().tolist()

        print("=*="*10)
        print("start test case : ")
        print("=*="*10)

        print("-&"*10)
        print(idx)
        print("doc_idx: ", tmp_doc_idx[0])
        print("gold_mention_span: ", tmp_mention_span)
        print("gold_cluster_ids:", tmp_cluster_ids)


        coref_prediction_dict[tmp_doc_idx[0]] = tmp_cluster_ids 
        coref_evaluator.update(tmp_cluster_ids, tmp_cluster_ids, tmp_mention_span, tmp_mention_span)

        if idx > 5:
            break

    summary_dict = {}
    conll_results = conll.evaluate_conll(config.eval_path, coref_prediction_dict, False)
    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    summary_dict["Average F1 (conll)"] = average_f1 
    print("Average F1 (conll) : {:.2f}%".format(average_f1))

    p, r, f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f 
    print("Average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Average precision (py)"] = p 
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r 
    print("Average recall (py): {:.2f}%".format(r * 100))
    print("###"*20)




