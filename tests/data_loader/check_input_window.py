#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


import re 
import os 
import sys 


REPO_PATH = '/'.join(os.path.realpath(__file__).split("/")[:-3])
print(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


from data_loader.conll_data_processor import parse_document, flatten_clusters, read_conll_file, tokenize_document, construct_sliding_windows, expand_with_speakers
from transformers.tokenization import BertTokenizer 


SPEAKER_START = '[unused19]'
SPEAKER_END = '[unused73]'



if __name__ == "__main__":
    conll_file_path = "/xiaoya/data/test.english.v4_gold_conll"
    data_instances = read_conll_file(conll_file_path)
    vocab_file = os.path.join(REPO_PATH, "data_preprocess", "vocab.txt")
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)
    # print(len(data_instances))
    # print(data_instances[0])
    data_example = data_instances[0]
    doc_info = parse_document(data_example, 'english')

    tokenized_document = tokenize_document(doc_info, tokenizer)
    expand_token, expand_mask = expand_with_speakers(tokenized_document)
    print(expand_token, expand_mask) 

    sliding_window_size = 50 
    sliding_windows = construct_sliding_windows(len(expand_token), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expand_token[window_start: window_end]
        original_masks = expand_mask[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = ['[CLS]'] + original_tokens + ['[SEP]'] + ['[PAD]'] * (sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        print(one_window_mask)
        print("-*-"*10)
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)

    token_windows = token_windows 
    mask_windows = mask_windows 
    span_start, span_end, mention_span, cluster_ids = flatten_clusters(tokenized_document['clusters'])
    print("span_start", span_start)
    print("span_end", span_end)
    print("mention_span", mention_span)
    print("clutster_ids", cluster_ids)
    print("-*-"*10)
    print("check the mention in the subword list: ")
    for tmp_start, tmp_end in zip(span_start, span_end):
        tmp_start_idx = expand_mask.index(tmp_start)
        tmp_end_idx = expand_mask.index(tmp_end)
        print(expand_token[tmp_start_idx: tmp_end_idx+1])









