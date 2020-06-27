# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: test_corefqa
@time: 2020/6/23 20:37

"""

import os
import torch
from model.corefqa import CorefQA
from data_loader.conll_dataloader import CoNLLDataLoader, CoNLLDataset
from transformers import BertConfig

bert_model = "/dev/shm/xiaoya/pretrain_ckpt/cased_L-12_H-768_A-12"
bert_config = BertConfig.from_json_file(os.path.join(bert_model, "config.json"))



config_dict = {'bert_config_file': '/data/nfsdata/nlp/BERT_BASE_DIR/cased_L-12_H-768_A-12/bert_config.json',
                    'span_ratio': 0.2, 'max_candidate_num': 200, 'max_antecedent_num': 30, 'max_span_width': 5,
                    'sliding_window_size': 128, 'pad_idx': -4, 'use_span_width_embeddings': False,
                    'span_width_embed_size': 1024, 'ffnn_size': 1024, 'ffnn_depth': 1, 'mention_start_idx': 37,
                    'mention_end_idx': 42, 'speaker_start_idx': 19, 'speaker_end_idx': 73, 'max_question_len': 128,
                    'num_epochs': 10, 'num_docs': 2802, 'save_checkpoints_steps': 500,
                    'init_checkpoint': '/dev/shm/xiaoya/wuwei_spanbert/model.ckpt',
                    'train_file': '/data/nfs/wuwei/study/CorefQA/data/train.english.tfrecord',
                    'dev_conll_file': '/dev/shm/xiaoya/data/dev.english.v4_gold_conll',
                    'test_conll_file': '/dev/shm/xiaoya/data/test.english.v4_gold_conll', 'mention_threshold': 0.5,
                    'mention_loss_ratio': 0.7, 'bert_init_lr': 1e-05, 'task_init_lr': 1e-05,
                    'bert_opt': 'adam_weight_decay', 'task_opt': 'adam_weight_decay', 'freeze': -1, 'warmup_ratio': 0.1,
                    'config_path': '/home/lixiaoya/yuxian/coref/config/gpu_spanbert.yml',
                    'config_name': 'spanbert_base', 'data_dir': '/dev/shm/xiaoya/data',
                    'bert_model': '/dev/shm/xiaoya/yuxian/corefqa_pytorch_output/2020.06.24_morn', 'do_eval': True,
                    'lr': 3e-05, 'eval_per_epoch': 3, 'warmup_proportion': -1.0, 'num_train_epochs': 200,
                    'local_rank': -1, 'gradient_accumulation_steps': 1, 'seed': 2333, 'n_gpu': 1,
                    'fp16_opt_level': 'O3',
                    'output_dir': '/dev/shm/xiaoya/yuxian/corefqa_pytorch_output/2020.06.24_morn', 'data_cache': False,
                    'fp16': False, 'loss_scale': 1.0, 'tpu': False, 'debug': False, 'mention_chunk_size': 1,
                    'mention_proposal_only': False, 'use_cache_data': False, 'save_model': True, 'train_batch_size': 1,
                    'is_padding': True
               }

class Config:
    def __init__(self):
        pass

config = Config()

for key, value in config_dict.items():
    setattr(config, key, value)


device = 0
MODEL: CorefQA = CorefQA(bert_config, config, device)


def test_mention_proposal():
    """test forward"""
    window_input_ids = torch.LongTensor([[200, 200, 200, 300, 300]]).long()  # [num_window, ]
    sentence_map = torch.LongTensor([0, 0, 0, 1, 1])
    window_masked_ids = torch.LongTensor([0, 1, 2, 3, 4])
    span_starts = torch.LongTensor([0])
    span_ends = torch.LongTensor([1])
    cluster_ids = torch.LongTensor([1])
    y = MODEL(sentence_map=sentence_map, subtoken_map=None,
              window_input_ids=window_input_ids, window_masked_ids=window_masked_ids,
              gold_mention_span=None, token_type_ids=None, attention_mask=None,
              span_starts=span_starts, span_ends=span_ends, cluster_ids=cluster_ids)
    print(y)


def test_get_candidates_spans():
    sentence_map = torch.Tensor([0, 0, 1, 1, 2, 2]).long()
    doc_ids = torch.Tensor([1, 2, 3, 4, 0, 0]).long()
    starts, ends, mask = MODEL.get_candidate_spans(sentence_map, doc_ids=doc_ids)
    golden_starts = torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]).long()
    golden_ends = torch.Tensor([0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 5, 3, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5]).long()
    golden_mask = torch.Tensor([True, True, False, False, False, True, False, False, False, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]).long()
    assert ((starts == golden_starts).all().item() and
            (ends == golden_ends).all().item() and
            (mask == golden_mask).all().item())


def test_get_question_tokens():
    sentence_map = torch.Tensor([0, 0, 0, 1, 1]).long()
    flattened_input_ids = torch.Tensor([0, 1, 2, 3, 4]).long()
    flattened_input_mask = torch.Tensor([0, 1, 2, 3, 4]).long()
    span_start = 0
    span_end = 1
    question_tokens = MODEL.get_question_token_ids(sentence_map=sentence_map,
                                                   flattened_input_ids=flattened_input_ids,
                                                   flattened_input_mask=flattened_input_mask,
                                                   span_start=span_start,
                                                   span_end=span_end)
    golden_question_tokens = torch.Tensor([MODEL.mention_start_idx, 0, 1, MODEL.mention_end_idx, 2]).long()
    assert (golden_question_tokens == question_tokens).all().item()


def test_fast_get_question_tokens():
    sentence_map = torch.Tensor([0, 0, 0, 1, 1]).long()
    doc_ids = torch.Tensor([0, 1, 2, 3, 4]).long()
    span_start = 0
    span_end = 1
    question_tokens, start, end = MODEL.fast_get_question_token_ids(
        sentence_map=sentence_map,
        doc_ids=doc_ids,
        span_start=span_start,
        span_end=span_end,
        return_offset=True
    )
    golden_start, golden_end = 1, 2
    golden_question_tokens = torch.Tensor([MODEL.mention_start_idx, 0, 1, MODEL.mention_end_idx, 2]).long()
    assert (golden_question_tokens == question_tokens).all().item()
    assert start.item() == golden_start and end.item() == golden_end


def test_pad():
    tensors = [torch.LongTensor([[1, 2, 3], [1, 2, 3]]),
               torch.LongTensor([[1, 2], [1, 2]]),
               ]
    padded_tensor = CorefQA.pad_stack(tensors, 1)
    golden_tensor = torch.LongTensor(
        [
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 1], [1, 2, 1]]
        ]
    )
    assert (padded_tensor == golden_tensor).all().item()


def test_forward_with_conll_data():
    """test forward"""

    class Config(object):
        def __init__(self, ):
            self.data_dir = "/dev/shm/xiaoya/data"
            self.sliding_window_size = 128

    config = Config()
    print(config.data_dir)
    dataloader = CoNLLDataLoader(config)
    test_dataloader = dataloader.get_dataloader("test")
    for test_example in test_dataloader:
        print("=*=" * 10)

        (proposal_loss, sentence_map, input_ids, masked_input_ids,
         candidate_starts, candidate_ends, candidate_labels, candidate_mention_scores,
         topk_span_starts, topk_span_ends, topk_span_labels, topk_mention_scores) = MODEL(
            sentence_map=test_example["sentence_map"].squeeze(0),
            subtoken_map=None,
            window_input_ids=test_example["flattened_input_ids"].view(-1, config.sliding_window_size),
            window_masked_ids=test_example["flattened_input_mask"].view(-1, config.sliding_window_size),
            gold_mention_span=test_example["mention_span"].squeeze(0),
            token_type_ids=None, attention_mask=None,
            span_starts=test_example["span_start"].squeeze(0),
            span_ends=test_example["span_end"].squeeze(0),
            cluster_ids=test_example["cluster_ids"].squeeze(0),
        )

        link_loss = MODEL.batch_qa_linking(
            sentence_map=sentence_map,
            window_input_ids=input_ids,
            window_masked_ids=masked_input_ids,
            token_type_ids=None,
            attention_mask=None,
            candidate_starts=candidate_starts,
            candidate_ends=candidate_ends,
            candidate_labels=candidate_labels,
            candidate_mention_scores=candidate_mention_scores,
            topk_span_starts=topk_span_starts,
            topk_span_ends=topk_span_ends,
            topk_span_labels=topk_span_labels,
            topk_mention_scores=topk_mention_scores,
            origin_k=topk_span_starts.shape[0],
            gold_mention_span=None,
            recompute_mention_scores=True
        )

        print(link_loss)
