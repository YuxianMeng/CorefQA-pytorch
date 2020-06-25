#!/usr/bin/env python3 
# -*- utf-8 -*- 



# author: xiaoy li 
# description:
# dataloader for conll-2012 


import os 
import torch
from typing import List


from torch.utils.data import DataLoader, SequentialSampler, Dataset
from data_loader.conll_data_processor import prepare_conll_dataset, CoNLLCorefResolution


class CoNLLDataset(Dataset):
    def __init__(self, features: List[CoNLLCorefResolution]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature: CoNLLCorefResolution = self.features[item]
        return {
            "doc_idx": torch.tensor([feature.doc_idx], dtype=torch.int64),
            "sentence_map": torch.tensor(feature.sentence_map, dtype=torch.int64),
            "subtoken_map": torch.tensor(feature.subtoken_map, dtype=torch.int64),
            "flattened_input_ids": torch.tensor(feature.flattened_input_ids, dtype=torch.int64),
            "flattened_input_mask": torch.tensor(feature.flattened_input_mask, dtype=torch.int64),
            "span_start": torch.tensor(feature.span_start, dtype=torch.int64),
            "span_end": torch.tensor(feature.span_end, dtype=torch.int64),
            "cluster_ids": torch.tensor(feature.cluster_ids, dtype=torch.int64),
            "mention_span": torch.tensor(feature.mention_span, dtype=torch.int64)
        }


class CoNLLDataLoader(object):
    def __init__(self, config, tokenizer=None, mode="train", language="english"):
        self.data_dir = config.data_dir 
        self.language = language 
        self.sliding_window_size = config.sliding_window_size
        self.config = config 

        if mode == "train":
            self.train_batch_size = 1 
            self.dev_batch_size = 1
            self.test_batch_size = 1 
        else:
            self.test_batch_size = 1 

        self.tokenizer = tokenizer 

        self.num_train_instance = 0 
        self.num_dev_instance = 0 
        self.num_test_instance = 0 

    def convert_examples_to_features(self, data_sign="train"):
        input_file = os.path.join(self.data_dir, "{}.{}.v4_gold_conll".format(data_sign, self.language))
        features = prepare_conll_dataset(input_file, self.sliding_window_size, )
        self.__setattr__("num_{}_instance".format(data_sign), len(features))

        return features

    def get_dataloader(self, data_sign="train"):

        features = self.convert_examples_to_features(data_sign=data_sign)
        dataset = CoNLLDataset(features)

        if data_sign == "dev":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.dev_batch_size)
        elif data_sign == "test":
            datasampler = SequentialSampler(dataset) 
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)
        else:
            datasampler = SequentialSampler(dataset) # RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)

        return dataloader 

