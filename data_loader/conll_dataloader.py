#!/usr/bin/env python3 
# -*- utf-8 -*- 



# author: xiaoy li 
# description:
# dataloader for conll-2012 


import os 
import torch 


from torch.utils.data import TensorDataset, DataLoader, SequentialSampler  
from data_loader.conll_data_processor import prepare_conll_dataset



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

        if data_sign == "train":
            input_file = os.path.join(self.data_dir, "train.{}.v4_gold_conll".format(self.language))
            features = prepare_conll_dataset(input_file, self.sliding_window_size, )
            self.num_train_instance = len(features)
        elif data_sign == "dev":
            input_file = os.path.join(self.data_dir, "dev.{}.v4_gold_conll".format(self.language))
            features = prepare_conll_dataset(input_file, self.sliding_window_size, )
            self.num_dev_instance = len(features)
        elif data_sign == "test":
            input_file = os.path.join(self.data_dir, "test.{}.v4_gold_conll".format(self.language))
            features = prepare_conll_dataset(input_file, self.sliding_window_size, )
            self.num_test_instance = len(features)
        else:
            raise ValueError 

        return features


    def get_dataloader(self, data_sign="train"):

        features = self.convert_examples_to_feautres(data_sign=data_sign)

        doc_idx = torch.tensor([f.doc_idx for f in features], dtype=torch.long)
        sentence_map = torch.tensor([f.sentence_map for f in features], dtype=torch.long)
        subtoken_map = torch.tensor([f.subtoken_map for f in features], dtype=torch.long)
        flattened_input_ids = torch.tensor([f.flattened_input_ids for f in features], dtype=torch.long)
        flattened_input_mask = torch.tensor([f.flattened_input_mask for f in features], dtype=torch.long)
        span_start = torch.tensor([f.span_start for f in features], dtype=torch.long)
        span_end = torch.tensor([f.span_end for f in features], dtype=torch.long)
        mention_span = torch.tensor([f.mention_span for f in features], dtype=torch.long)
        cluster_ids = torch.tensor([f.cluster_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(doc_idx, sentence_map,subtoken_map, flattened_input_ids, flattened_input_mask, \
            span_start, span_end, mention_span, cluster_ids )


        if data_sign == "train":
            datasampler = SequentialSampler(dataset) # RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)
        elif data_sign == "dev":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.dev_batch_size)
        elif data_sign == "test":
            datasampler = SequentialSampler(dataset) 
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)

        return dataloader 

















