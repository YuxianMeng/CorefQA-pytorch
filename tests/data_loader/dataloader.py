#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# author: xiaoy li 
# description:
# test data loader


import os 
import sys 


REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-3])
print(REPO_PATH)

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


from data_loader.conll_dataloader import CoNLLDataLoader


class Config(object):
    def __init__(self, ):
        self.data_dir = "/dev/shm/xiaoya/data"
        self.sliding_window_size = 128


if __name__ == "__main__":
    config = Config()
    print(config.data_dir)
    dataloader = CoNLLDataLoader(config)

    # flattened_input_ids 
    # sentece_map 

    loader = dataloader.get_dataloader("test")
    for idx, test_example in enumerate(loader):
        print("=*"*10)
        # print(test_example)
        # print(test_example["doc_idx"].squeeze(0).numpy().tolist())
        print("sentence_map: ", len(test_example["sentence_map"].squeeze(0).numpy().tolist()), test_example["sentence_map"].squeeze(0).numpy().tolist())
        print("subtoken_map: ", len(test_example["subtoken_map"].squeeze(0).numpy().tolist()), test_example["subtoken_map"].squeeze(0).numpy().tolist())
        print("flattened_input_ids: ", len(test_example["flattened_input_ids"].squeeze(0).numpy().tolist()[0]), test_example["flattened_input_ids"].squeeze(0).numpy().tolist()[0])
        print("flattened_input_mask: ", len(test_example["flattened_input_mask"].squeeze(0).numpy().tolist()[0]), test_example["flattened_input_mask"].squeeze(0).numpy().tolist()[0])
        print(test_example["span_start"].squeeze(0).numpy().tolist())
        print(test_example["span_end"].squeeze(0).numpy().tolist())
        print(test_example["mention_span"].squeeze(0).numpy().tolist())
        print(test_example["cluster_ids"].squeeze(0).numpy().tolist())

        if idx == 0:
            break







