#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# test evaluate metrics 




import os 
import sys 
import json 


REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-3])
print(REPO_PATH)

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


import data_preprocess.conll as conll 
# clusters: [[[21, 25], [18, 18]], [[61, 63], [44, 46], [27, 29]], [[87, 87], [86, 86]]]
# doc_key: bc/cctv/00/cctv_0005_0


class Config(object):
    def __init__(self, ):
        self.data_dir = "/dev/shm/xiaoya/data"
        self.sliding_window_size = 128
        self.eval_path = os.path.join(self.data_dir, "test.english.v4_gold_conll")
        self.eval_json = os.path.join(self.data_dir, "test.english.512.jsonlines")




if __name__ == "__main__":
    config = Config()
    print(config.data_dir)
    eval_file_path = config.eval_json 
    coref_prediction_dict = {}
    subtoken_map_dict = {}

    with open(eval_file_path, "r") as f:
        data_instances = f.readlines()
        for data_idx, data_item in enumerate(data_instances):
            data_item_dict = json.loads(data_item)
            coref_prediction_dict[data_item_dict["doc_key"]] = data_item_dict["clusters"]
            subtoken_map_dict[data_item_dict["doc_key"]] = data_item_dict["subtoken_map"]
            print("-*-"*10)
            print("doc_key: ", data_item_dict["doc_key"])
            print("subtoken_map: ", data_item_dict["subtoken_map"])
            print("clusters: ", data_item_dict["clusters"])


    conll_results = conll.evaluate_conll(config.eval_path, coref_prediction_dict, subtoken_map_dict, official_stdout=True)
    print(conll_results)


    
    









