#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# corefqa model 



import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 


from config.load_config import BertConfig
from module.classifier import MultiNonLinearClassifier 

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel 



class CorefQA(BertPreTrainedModel):
    def __init__(self, config):
        super(CorefQA, self).__init__(config)

        self.config = config 
        self.bert_config = BertConfig.from_json_file(self.config.bert_config_file)

        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)
        # mention proposal 
        self.mention_start_ffnn = nn.Linear(self.bert_config.hidden_size, 1)
        self.mention_end_ffnn = nn.Linear(self.bert_config.hidden_size, 1)
        self.mention_span_ffnn = nn.Linear(self.bert_config.hidden_size*2, 1)

        # cluster 
        self.forward_qa_ffnn = nn.Linear(self.bert_config.hidden_size*2, 1)
        self.backward_qa_ffnn = nn.Linear(self.bert_config.hidden_size*2, 1)

    def forward(self, doc_idx, sentence_map, subtoken_map, input_ids, \
            token_type_ids=None, attention_mask=None, span_start=None, span_end=None, cluster_ids=None):
        
        mention_sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        




        return prediction, loss 






















