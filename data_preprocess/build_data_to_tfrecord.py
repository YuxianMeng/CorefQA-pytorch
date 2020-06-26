#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Wei Wu
@license: Apache Licence
@file: prepare_training_data.py
@time: 2019/12/19
@contact: wu.wei@pku.edu.cn

将conll的v4_gold_conll文件格式转成模型训练所需的jsonlines数据格式
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import List, Tuple

import tensorflow as tf

import data_preprocess.conll as conll 
from bert.tokenization import FullTokenizer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, help='The data directory to save *.v4_gold_conll and *.jsonlines')
parser.add_argument('--language', type=str, default='english', help='The language to process.')
parser.add_argument('--vocab_file', type=str, help='The original vocab file for bert tokenization')
parser.add_argument('--sliding_window_size', type=int, default=384, help='Sliding window size for BERT processing')
args = parser.parse_args()

handlers = [logging.FileHandler(filename='prepare_training_data.log'), logging.StreamHandler(sys.stdout)]
logging.basicConfig(level=logging.INFO, handlers=handlers,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logger = logging.getLogger()

SPEAKER_START = '[unused19]'
SPEAKER_END = '[unused73]'


def read_conll_file(conll_file_path: str) -> List[Tuple]:
    documents = []
    with open(conll_file_path) as fi:
        for line in fi:
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line.strip())
    return documents


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def parse_document(document: Tuple[str, List], language: str) -> dict:
    """
    get basic information from one document annotation.
    :param document:
    :param language: english, chinese or arabic
    :return:
    """
    doc_key = document[0]
    sentences = [[]]
    speakers = []
    coreferences = []
    word_idx = -1
    last_speaker = ''
    for line_id, line in enumerate(document[1]):
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            word_idx += 1
            word = normalize_word(row[3], language)
            sentences[-1].append(word)
            speaker = row[9]
            if speaker != last_speaker:
                speakers.append((word_idx, speaker))
                last_speaker = speaker
            coreferences.append(row[-1])
        else:
            sentences.append([])
    clusters = coreference_annotations_to_clusters(coreferences)
    doc_info = {'doc_key': doc_key, 'sentences': sentences[: -1], 'speakers': speakers, 'clusters': clusters}
    return doc_info


def coreference_annotations_to_clusters(annotations: List[str]) -> List[List[Tuple]]:
    """
    convert coreference information to clusters
    :param annotations:
    :return:
    """
    clusters = defaultdict(list)
    coref_stack = defaultdict(list)
    for word_idx, annotation in enumerate(annotations):
        if annotation == '-':
            continue
        for ann in annotation.split('|'):
            cluster_id = int(ann.replace('(', '').replace(')', ''))
            if ann[0] == '(' and ann[-1] == ')':
                clusters[cluster_id].append((word_idx, word_idx))
            elif ann[0] == '(':
                coref_stack[cluster_id].append(word_idx)
            elif ann[-1] == ')':
                span_start = coref_stack[cluster_id].pop()
                clusters[cluster_id].append((span_start, word_idx))
            else:
                raise NotImplementedError
    assert all([len(starts) == 0 for starts in coref_stack.values()])
    return list(clusters.values())


def checkout_clusters(doc_info):
    words = [i for j in doc_info['sentences'] for i in j]
    clusters = [[' '.join(words[start: end + 1]) for start, end in cluster] for cluster in doc_info['clusters']]
    print(clusters)


def tokenize_document(doc_info: dict, tokenizer: FullTokenizer) -> dict:
    """
    tokenize into sub tokens
    :param doc_info:
    :param tokenizer:
    :return:
    """
    sub_tokens: List[str] = []  # all sub tokens of a document
    sentence_map: List[int] = []  # collected tokenized tokens -> sentence id
    subtoken_map: List[int] = []  # collected tokenized tokens -> original token id
    word_idx = -1

    for sentence_id, sentence in enumerate(doc_info['sentences']):
        for token in sentence:
            word_idx += 1
            word_tokens = tokenizer.tokenize(token)
            sub_tokens.extend(word_tokens)
            sentence_map.extend([sentence_id] * len(word_tokens))
            subtoken_map.extend([word_idx] * len(word_tokens))

    speakers = {subtoken_map.index(word_index): tokenizer.tokenize(speaker)
                for word_index, speaker in doc_info['speakers']}
    clusters = [[(subtoken_map.index(start), len(subtoken_map) - 1 - subtoken_map[::-1].index(end))
                 for start, end in cluster] for cluster in doc_info['clusters']]
    tokenized_document = {'sub_tokens': sub_tokens, 'sentence_map': sentence_map, 'subtoken_map': subtoken_map,
                          'speakers': speakers, 'clusters': clusters, 'doc_key': doc_info['doc_key']}
    return tokenized_document


def expand_with_speakers(tokenized_document: dict) -> Tuple[List[str], List[int]]:
    """
    add speaker name information
    :param tokenized_document: tokenized document information
    :return:
    """
    expanded_tokens = []
    expanded_masks = []
    for token_idx, token in enumerate(tokenized_document['sub_tokens']):
        if token_idx in tokenized_document['speakers']:
            speaker = [SPEAKER_START] + tokenized_document['speakers'][token_idx] + [SPEAKER_END]
            expanded_tokens.extend(speaker)
            expanded_masks.extend([-1] * len(speaker))
        expanded_tokens.append(token)
        expanded_masks.append(token_idx)
    return expanded_tokens, expanded_masks


def construct_sliding_windows(sequence_length: int, sliding_window_size: int):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows


def flatten_clusters(clusters: List[List[Tuple[int, int]]]) -> Tuple[List[int], List[int], List[int]]:
    """
    flattern cluster information
    :param clusters:
    :return:
    """
    span_starts = []
    span_ends = []
    cluster_ids = []
    for cluster_id, cluster in enumerate(clusters):
        for start, end in cluster:
            span_starts.append(start)
            span_ends.append(end)
            cluster_ids.append(cluster_id + 1)
    return span_starts, span_ends, cluster_ids


def convert_to_sliding_window(tokenized_document: dict, sliding_window_size: int):
    """
    construct sliding windows, allocate tokens and masks into each window
    :param tokenized_document:
    :param sliding_window_size:
    :return:
    """
    expanded_tokens, expanded_masks = expand_with_speakers(tokenized_document)
    sliding_windows = construct_sliding_windows(len(expanded_tokens), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expanded_tokens[window_start: window_end]
        original_masks = expanded_masks[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = ['[CLS]'] + original_tokens + ['[SEP]'] + ['[PAD]'] * (sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    assert len(tokenized_document['sentence_map']) == sum([i >= 0 for j in mask_windows for i in j])
    return token_windows, mask_windows


def prepare_training_data(data_dir: str, language: str, vocab_file: str, sliding_window_size: int):
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    for dataset in ['train', 'dev', 'test']:
        conll_file_path = os.path.join(data_dir, F"{dataset}.{language}.v4_gold_conll")
        writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, F"{dataset}.{language}.tfrecord"))
        doc_map = {}
        documents = read_conll_file(conll_file_path)
        for doc_idx, document in enumerate(documents):
            doc_info = parse_document(document, language)
            checkout_clusters(doc_info)
            tokenized_document = tokenize_document(doc_info, tokenizer)
            doc_map[doc_idx] = tokenized_document['doc_key']
            token_windows, mask_windows = convert_to_sliding_window(tokenized_document, sliding_window_size)
            input_id_windows = [tokenizer.convert_tokens_to_ids(tokens) for tokens in token_windows]
            span_starts, span_ends, cluster_ids = flatten_clusters(tokenized_document['clusters'])
            instance = (doc_idx, tokenized_document['sentence_map'], tokenized_document['subtoken_map'],
                        input_id_windows, mask_windows, span_starts, span_ends, cluster_ids)
            write_instance_to_example_file(writer, instance)
        with open(os.path.join(data_dir, F"{dataset}.{language}.map"), 'w') as fo:
            json.dump(doc_map, fo, indent=2)


def write_instance_to_example_file(writer: tf.python_io.TFRecordWriter, instance: tuple):
    doc_idx, sentence_map, subtoken_map, input_id_windows, mask_windows, span_starts, span_ends, cluster_ids = instance
    flattened_input_ids = [i for j in input_id_windows for i in j]
    flattened_input_mask = [i for j in mask_windows for i in j]
    features = {
        'doc_idx': create_int_feature([doc_idx]),
        'sentence_map': create_int_feature(sentence_map),
        'subtoken_map': create_int_feature(subtoken_map),
        'flattened_input_ids': create_int_feature(flattened_input_ids),
        'flattened_input_mask': create_int_feature(flattened_input_mask),
        'span_starts': create_int_feature(span_starts),
        'span_ends': create_int_feature(span_ends),
        'cluster_ids': create_int_feature(cluster_ids),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


if __name__ == '__main__':
    prepare_training_data(args.data_dir, args.language, args.vocab_file, args.sliding_window_size)
