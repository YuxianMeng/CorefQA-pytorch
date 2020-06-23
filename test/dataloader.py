#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# test data loader


import os 
import sys 
REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from data_loader.conll_dataloader import CoNLLDataLoader 


class Config(object):
    def __init__(self, ):
        self.data_dir = "/xiaoya/data"
        self.sliding_window_size = 128


if __name__ == "__main__":
    config = Config()
    print(config.data_dir)
    dataloader = CoNLLDataLoader(config)

    test_features = dataloader.convert_examples_to_features(data_sign="test")
    test_example = test_features[0]
    print("=*="*10)
    print(test_example.doc_idx)
    print(test_example.sentence_map)
    print(test_example.subtoken_map)

    print(test_example.flattened_input_ids)
    print(test_example.flattened_input_mask)
    print(test_example.span_start)
    print(test_example.span_end)
    print(test_example.mention_span)
    print(test_example.cluster_ids)


