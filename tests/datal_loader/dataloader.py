#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# author: xiaoy li 
# description:
# test data loader


import os 
import sys 


from data_loader.conll_dataloader import CoNLLDataLoader, CoNLLDataset


class Config(object):
    def __init__(self, ):
        self.data_dir = "/xiaoya/data"
        self.sliding_window_size = 128


if __name__ == "__main__":
    config = Config()
    print(config.data_dir)
    dataloader = CoNLLDataLoader(config)

    # test_features = dataloader.convert_examples_to_features("test")
    # test_example = test_features[0]
    # print("=*="*10)
    # print(test_example.doc_idx)
    # print(test_example.sentence_map)
    # print(test_example.subtoken_map)
    #
    # print(test_example.flattened_input_ids)
    # print(test_example.flattened_input_mask)
    # print(test_example.span_start)
    # print(test_example.span_end)
    # print(test_example.mention_span)
    # print(test_example.cluster_ids)

    loader = dataloader.get_dataloader("test")
    for idx, t in enumerate(loader):
        print(t)
        if idx > 5:
            break
