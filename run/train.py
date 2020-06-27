#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# author: xiaoy li
# description:
# 


import os
import yaml
import time
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm

import data_preprocess.conll as conll
from config.load_config import Config
from transformers.modeling import BertConfig
from data_loader.conll_dataloader import CoNLLDataLoader
from model.corefqa import CorefQA
from module import metrics
from module.optimization import AdamW, warmup_linear
from module import model_utils
from utils.logger import get_logger

LOGGING = get_logger(__name__)

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except:
    LOGGING.info("=*=" * 10)
    LOGGING.info("DONOT ON TPU Now !!! ")
    LOGGING.info("IMPORT torch_xla when running on the Google Cloud TPU Pod. ")
    LOGGING.info("=*=" * 10)


def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters 
    parser.add_argument("--config_path", default="/home/lixiaoya/bert.yaml", type=str)
    parser.add_argument("--config_name", default="spanbert_base", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default=None, type=str, )
    parser.add_argument("--do_eval", default=False, type=bool)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--eval_per_epoch", default=10, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--fp16_opt_level", type=str, default="O3",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/output")
    parser.add_argument("--data_cache", type=bool, default=False)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--loss_scale", type=float, default=1.0, )
    parser.add_argument('--tpu', action='store_true', help="Whether to use tpu machine")
    parser.add_argument('--debug', action='store_true', help="print some debug information.")
    parser.add_argument("--mention_chunk_size", type=int, default=1, help="use mention chunk to reduce memory usage")
    parser.add_argument("--mention_proposal_only", action="store_true", help="only train mention proposal network")
    parser.add_argument("--use_cache_data", action="store_true", help="use cache data to save time")
    parser.add_argument("--save_model", action="store_true", help="save model")
    parser.add_argument("--is_padding", action="store_false", help="pad doc to max_doc_length")

    args = parser.parse_args()

    args.train_batch_size = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_data(config, data_sign="conll"):
    if data_sign == "conll":
        dataloader = CoNLLDataLoader(config)
    else:
        raise ValueError(">>> DO NOT IMPLEMENT GAP DATASET >>>")

    train_dataloader = dataloader.get_dataloader("train", use_cache=config.use_cache_data)
    dev_dataloader = dataloader.get_dataloader("dev", use_cache=config.use_cache_data)
    test_dataloader = dataloader.get_dataloader("test", use_cache=config.use_cache_data)
    # train_dataloader = dataloader.get_dataloader("dev1", use_cache=config.use_cache_data)
    # dev_dataloader = dataloader.get_dataloader("dev1", use_cache=config.use_cache_data)
    # test_dataloader = dataloader.get_dataloader("dev1", use_cache=config.use_cache_data)

    return train_dataloader, dev_dataloader, test_dataloader


def load_model(config):
    if config.tpu:
        device = xm.xla_device()
        n_gpu = 0
    else:
        device = torch.device("cuda")
        LOGGING.info("-*-" * 10)
        LOGGING.info("please notice that the device is :")
        LOGGING.info(device)
        LOGGING.info("-*-" * 10)
        n_gpu = config.n_gpu
    # naive_bert_config = BertConfig.from_json_file(os.path.join(config.bert_model, "config.json"))
    # naive_model = CorefQA(naive_bert_config, config, device)
    model = CorefQA.from_pretrained(config.bert_model, config=config, device=device)

    model = model.to(device)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=10e-8)

    if config.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                              "to use distributed and fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    # if config.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True
    #     )

    sheduler = None
    return model, optimizer, sheduler, device, n_gpu


def backward_loss(optimizer, loss, fp16=False, retain_graph=False):
    if fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)


def train(model: CorefQA, optimizer, sheduler, train_dataloader, dev_dataloader, test_dataloader, config,
          device, n_gpu, ):
    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps = 0
    global_step = 0

    best_dev_average_f1 = 0
    best_dev_summary_dict = None

    test_average_f1_when_dev_best = 0
    test_summary_dict_when_dev_best = None

    start_time = time.time()
    num_train_optimization_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    train_batches = [batch for batch in train_dataloader]
    eval_step = max(1, len(train_batches) // config.eval_per_epoch)

    if config.do_eval and config.mention_proposal_only:
        LOGGING.info(f"evaluating before training")
        precision, recall, f1 = evaluate_mention_proposal(model=model, dataloader=dev_dataloader,
                                                          device=device)
        LOGGING.info(f"metrics of mention proposal before training: f1:{f1}, precision: {precision}, recall: {recall}")

    for epoch in range(int(config.num_train_epochs)):
        epoch_loss = 0.0
        LOGGING.info("=*=" * 20)
        LOGGING.info("start {} Epoch ... ".format(str(epoch)))
        model.train()
        LOGGING.info("Start epoch #{} (lr = {})...".format(epoch, config.lr))

        if config.debug:
            LOGGING.info("INFO: start train the CorefQA Model.")
        # if config.is_master:
        LOGGING.info(f'--- Starting epoch {epoch}/{int(config.num_train_epochs) - 1}')
        # if self.multi_gpu:
        #     torch.distributed.barrier()

        # iter_bar = tqdm(train_dataloader, desc="-Iter", disable=config.params.local_rank not in [-1, 0])
        iter_bar = tqdm(train_dataloader, desc="-Iter")
        for step, batch in enumerate(train_dataloader):
            ##if n_gpu == 1:
            ##    batch = tuple(t.to(device) for t in batch)
            sentence_map, subtoken_map, window_input_ids, window_masked_ids, gold_mention_span, token_type_ids, attention_mask, \
            span_starts, span_ends, cluster_ids = batch["sentence_map"].squeeze(0), None, \
                                                  batch["flattened_window_input_ids"].view(-1,
                                                                                           config.sliding_window_size), \
                                                  batch["flattened_window_masked_ids"].view(-1,
                                                                                            config.sliding_window_size), \
                                                  batch["mention_span"].squeeze(0), None, None, batch[
                                                      "span_start"].squeeze(0), batch["span_end"].squeeze(0), batch[
                                                      "cluster_ids"].squeeze(0)
            sentence_map = sentence_map.to(device)
            # subtoken_map = subtoken_map.to(device)
            window_input_ids = window_input_ids.to(device)
            window_masked_ids = window_masked_ids.to(device)
            gold_mention_span = gold_mention_span.to(device)
            span_starts = span_starts.to(device)
            span_ends = span_ends.to(device)
            cluster_ids = cluster_ids.to(device)

            if config.debug and step % 2 == 0:
                print("INFO: The {} epoch training process {}.".format(epoch, step))

            # loss = model(doc_idx=doc_idx, sentence_map=sentence_map, subtoken_map=subtoken_map, window_input_ids=window_input_ids, window_masked_ids=window_masked_ids, \
            #     gold_mention_span=gold_mention_span, token_type_ids=token_type_ids, attention_mask=attention_mask, span_starts=span_starts, span_ends=span_ends, cluster_ids=cluster_ids)

            loss = 0.0
            (proposal_loss, sentence_map, window_input_ids, window_masked_ids,
             candidate_starts, candidate_ends, candidate_labels, candidate_mention_scores,
             topk_span_starts, topk_span_ends, topk_span_labels, topk_mention_scores) = model(
                sentence_map=sentence_map,
                subtoken_map=subtoken_map,
                window_input_ids=window_input_ids,
                window_masked_ids=window_masked_ids,
                gold_mention_span=gold_mention_span,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                span_starts=span_starts,
                span_ends=span_ends,
                cluster_ids=cluster_ids
            )
            print(span_starts.shape, window_input_ids.shape)
            proposal_loss /= config.gradient_accumulation_steps
            tr_loss += proposal_loss.detach()
            if config.mention_chunk_size:
                backward_loss(optimizer=optimizer, fp16=config.fp16, loss=proposal_loss)
            else:
                loss += proposal_loss
            item_loss = 0
            item_loss += proposal_loss.detach()
            #
            # # use golden proposal to check code
            # topk_span_starts = span_starts
            # topk_span_ends = span_ends
            # topk_span_labels = cluster_ids
            # topk_mention_scores = torch.zeros_like(topk_span_starts)

            # mention linking
            # print(len(set(topk_span_starts.tolist()) & set(span_starts.tolist())), span_starts.shape[0])
            if not config.mention_proposal_only:
                mention_num = topk_span_starts.shape[0]
                chunk_num = mention_num // config.mention_chunk_size
                for chunk_idx in range(chunk_num):
                    chunk_start = config.mention_chunk_size * chunk_idx
                    chunk_end = chunk_start + config.mention_chunk_size
                    link_loss = model.batch_qa_linking(
                        sentence_map=sentence_map,
                        window_input_ids=window_input_ids,
                        window_masked_ids=window_masked_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        candidate_starts=candidate_starts,
                        candidate_ends=candidate_ends,
                        candidate_labels=candidate_labels,
                        candidate_mention_scores=candidate_mention_scores,
                        topk_span_starts=topk_span_starts[chunk_start: chunk_end],
                        topk_span_ends=topk_span_ends[chunk_start: chunk_end],
                        topk_span_labels=topk_span_labels[chunk_start: chunk_end],
                        topk_mention_scores=topk_mention_scores[chunk_start: chunk_end],
                        origin_k=mention_num,
                        gold_mention_span=gold_mention_span,
                        recompute_mention_scores=True
                    )
                    link_loss /= chunk_num
                    link_loss /= config.gradient_accumulation_steps
                    if config.mention_chunk_size:
                        backward_loss(optimizer=optimizer, loss=link_loss, fp16=config.fp16, retain_graph=False)
                    else:
                        loss += link_loss
                    item_loss += link_loss.detach()
                    tr_loss += link_loss.detach()
            epoch_loss += item_loss

            nb_tr_examples += window_input_ids.size(0)
            nb_tr_steps += 1

            if config.tpu:
                xm.optimizer_step(optimizer, barrier=True)

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.fp16:
                    lr_this_step = config.lr * warmup_linear(global_step / num_train_optimization_steps,
                                                             config.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % eval_step == 0:
                LOGGING.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                    epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))

                if config.do_eval:
                    if config.mention_proposal_only:
                        precision, recall, f1 = evaluate_mention_proposal(model=model, dataloader=dev_dataloader,
                                                                          device=device)
                        LOGGING.info(f"dev f1: {f1}, dev_p: {precision}, dev_r: {recall}")
                        if f1 > best_dev_average_f1:
                            LOGGING.info(f"best dev f1: {f1}, best_dev_p: {precision}, best_dev_r: {recall}")
                            best_dev_average_f1 = f1
                            if config.save_model:
                                output_model_file = os.path.join(config.output_dir,
                                                                 "{}_{}.checkpoint".format(str(epoch), str(step + 1)))
                                torch.save(model.state_dict(), output_model_file)
                                LOGGING.info(f"save model to {output_model_file}")
                            test_p, test_r, test_f1 = evaluate_mention_proposal(model=model, dataloader=test_dataloader,
                                                                                device=device)
                            LOGGING.info(f"test f1: {test_f1}, test_p: {test_p}, test_r: {test_r},")

                    else:
                        dev_summary_eval_dict, dev_average_f1 = evaluate(config, model, device, dev_dataloader, n_gpu,
                                                                         eval_sign="dev")

                        if dev_average_f1 > best_dev_average_f1:
                            best_dev_average_f1 = dev_average_f1
                            best_dev_summary_dict = dev_summary_eval_dict

                            if config.save_model:
                                model_to_save = model.module if hasattr(model, "module") else model
                                output_model_file = os.path.join(config.output_dir,
                                                                 "{}_{}.checkpoint".format(str(epoch), str(step + 1)))
                                # output_config_file = os.path.join(config.output_dir, "{}_{}.conf".format(str(epoch), str(step+1)))
                                torch.save(model_to_save.state_dict(), output_model_file)
                                LOGGING.info(f"save model to {output_model_file}")
                                # tokenizer.save_vocabulary(config.output_dir)

                            test_summary_dict_when_dev_best, test_average_f1_whem_dev_best = evaluate(config, model,
                                                                                                      device,
                                                                                                      test_dataloader,
                                                                                                      n_gpu,
                                                                                                      eval_sign="test")

                            with open(os.path.join(config.output_dir, "eval_results.txt"), "w") as writer:
                                writer.write("Dev RESULTS: \n")
                                for key in sorted(best_dev_summary_dict.keys()):
                                    writer.write("Dev: %s = %s\n" % (key, str(best_dev_summary_dict[key])))
                                writer.write("DEV Average (conll) F1 : %s" % (str(best_dev_average_f1)))
                                writer.write("=" * 10 + "\n")
                                writer.write("Test RESULTS: \n")
                                for key in sorted(test_summary_dict_when_dev_best.keys()):
                                    writer.write("Test: %s = %s\n" % (key, str(test_summary_dict_when_dev_best[key])))
                                writer.write("TEST Average (conll) F1 : %s" % (str(test_average_f1_when_dev_best)))
            iter_bar.update()
            # current_lr = sheduler.get_lr()[0]
            iter_bar.set_postfix({
                'loss_cur': f'{item_loss:.3f}',
                'loss_epoch': f'{epoch_loss / (step + 1):.3f}'
            })
        iter_bar.close()


def evaluate_mention_proposal(model: CorefQA, dataloader, device):
    """evaluate mention proposal network"""
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    epsilon = 1e-10
    for step, batch in enumerate(dataloader):
        sentence_map, subtoken_map, window_input_ids, window_masked_ids, gold_mention_span, token_type_ids, attention_mask, \
        span_starts, span_ends, cluster_ids = batch["sentence_map"].squeeze(0), None, \
                                              batch["flattened_window_input_ids"].view(-1, model.sliding_window_size), \
                                              batch["flattened_window_masked_ids"].view(-1, model.sliding_window_size), \
                                              batch["mention_span"].squeeze(0), None, None, batch["span_start"].squeeze(
            0), \
                                              batch["span_end"].squeeze(0), batch["cluster_ids"].squeeze(0)
        sentence_map = sentence_map.to(device)
        # subtoken_map = subtoken_map.to(device)
        window_input_ids = window_input_ids.to(device)
        window_masked_ids = window_masked_ids.to(device)
        gold_mention_span = gold_mention_span.to(device)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        cluster_ids = cluster_ids.to(device)

        with torch.no_grad():
            (proposal_loss, sentence_map, window_input_ids, window_masked_ids,
             candidate_starts, candidate_ends, candidate_labels, candidate_mention_scores,
             topk_span_starts, topk_span_ends, topk_span_labels, topk_mention_scores) = model(
                sentence_map=sentence_map,
                subtoken_map=subtoken_map,
                window_input_ids=window_input_ids,
                window_masked_ids=window_masked_ids,
                gold_mention_span=gold_mention_span,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                span_starts=span_starts,
                span_ends=span_ends,
                cluster_ids=cluster_ids
            )
        predict_labels = torch.sigmoid(candidate_mention_scores) > 0.5
        gold_labels = candidate_labels
        tp += torch.logical_and(predict_labels == 1, gold_labels == 1).sum()
        fp += torch.logical_and(predict_labels == 1, gold_labels == 0).sum()
        fn += torch.logical_and(predict_labels == 0, gold_labels == 1).sum()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, f1


def evaluate(config, model_object, device, dataloader, n_gpu, eval_sign="dev", official_stdout=False):
    model_object.eval()

    print("###" * 20)
    print("=" * 8 + "Evaluate {} dataset".format(eval_sign) + "=" * 8)

    coref_prediction = {}
    subtoken_map_dict = {}
    coref_evaluator = metrics.CorefEvaluator()
    eval_conll_file_path = config.dev_conll_path if eval_sign == "dev" else config.test_conll_path

    # top_span_starts, top_span_ends, predicted_antecedents, predicted_clusters
    for case_idx, case_feature in enumerate(dataloader):
        case_doc_key = dataloader.dataset.features[case_idx].doc_idx

        sentence_map, subtoken_map, window_input_ids, window_masked_ids, gold_mention_span, token_type_ids, attention_mask, \
        span_starts, span_ends, cluster_ids = case_feature["sentence_map"].squeeze(0), None, \
                                              case_feature["flattened_window_input_ids"].view(-1,
                                                                                              config.sliding_window_size), \
                                              case_feature["flattened_window_masked_ids"].view(-1,
                                                                                               config.sliding_window_size), \
                                              case_feature["mention_span"].squeeze(0), None, None, case_feature[
                                                  "span_start"].squeeze(0), case_feature["span_end"].squeeze(0), \
                                              case_feature["cluster_ids"].squeeze(0)
        sentence_map = sentence_map.to(device)
        # subtoken_map = subtoken_map.to(device)
        window_input_ids = window_input_ids.to(device)
        window_masked_ids = window_masked_ids.to(device)
        gold_mention_span = gold_mention_span.to(device)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        cluster_ids = cluster_ids.to(device)

        with torch.no_grad():
            (proposal_loss, sentence_map, window_input_ids, window_masked_ids, \
             candidate_starts, candidate_ends, candidate_labels, candidate_mention_scores, \
             topk_span_starts, topk_span_ends, topk_span_labels, topk_mention_scores) = model_object(
                sentence_map=sentence_map, subtoken_map=subtoken_map, window_input_ids=window_input_ids, \
                window_masked_ids=window_masked_ids, gold_mention_span=gold_mention_span, token_type_ids=token_type_ids, \
                attention_mask=attention_mask, span_starts=span_starts, span_ends=span_ends, \
                cluster_ids=cluster_ids, mode="eval")

            mention_num = topk_span_starts.shape[0]
            chunk_num = max(1, mention_num // config.mention_chunk_size)
            all_chunk_lst = []
            for chunk_idx in range(chunk_num):
                chunk_start = config.mention_chunk_size * chunk_idx
                chunk_end = chunk_start + config.mention_chunk_size
                link_loss, loss_antecedent_scores, mention_to_predict, mention_to_gold = model_object.batch_qa_linking(
                    sentence_map=sentence_map,
                    window_input_ids=window_input_ids,
                    window_masked_ids=window_masked_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    candidate_starts=candidate_starts,
                    candidate_ends=candidate_ends,
                    candidate_labels=candidate_labels,
                    candidate_mention_scores=candidate_mention_scores,
                    topk_span_starts=topk_span_starts[chunk_start: chunk_end],
                    topk_span_ends=topk_span_ends[chunk_start: chunk_end],
                    topk_span_labels=topk_span_labels[chunk_start: chunk_end],
                    topk_mention_scores=topk_mention_scores[chunk_start: chunk_end],
                    origin_k=mention_num,
                    gold_mention_span=gold_mention_span,
                    recompute_mention_scores=True,
                    mode="eval"
                )

                # loss_antecedent_scores: chunk-size * c+1
                all_chunk_lst.append(loss_antecedent_scores)

            all_chunk_antecedent_scores = torch.cat(all_chunk_lst, 0)
            predicted_antecedents = torch.argmax(all_chunk_antecedent_scores, dim=-1)

            predicted_antecedents = predicted_antecedents.detach().cpu().numpy().tolist()
            topk_span_starts = topk_span_starts.detach().cpu().numpy().tolist()
            topk_span_ends = topk_span_ends.detach().cpu().numpy().tolist()

            predicted_clusters, mention_to_predicted = model_utils.get_predicted_clusters(topk_span_starts,
                                                                                          topk_span_ends,
                                                                                          predicted_antecedents)

            mention_to_gold = mention_to_gold.to("cpu").numpy().tolist()

        coref_prediction[case_doc_key] = predicted_clusters
        subtoken_map_dict[case_doc_key] = case_feature["subtoken_map"]
        ## print("check the results : ")
        ## print(case_feature["subtoken_map"])
        ## print(predicted_clusters)
        ## exit()

        # coref_evaluator.update(pred_cluster_ids, gold_cluster_ids, mention_to_predict, mention_to_gold)

    summary_dict = {}
    conll_results = conll.evaluate_conll(eval_conll_file_path, coref_prediction, subtoken_map_dict, official_stdout)
    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    summary_dict["Average F1 (conll)"] = average_f1
    print("@" * 40)
    print("Average F1 (conll) : {:.2f}%".format(average_f1))
    print("CoNLL Results: ")
    print(conll_results)
    print("@" * 40)

    p, r, f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print("Average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Average precision (py)"] = p
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r
    print("Average recall (py): {:.2f}%".format(r * 100))
    print("###" * 20)

    return summary_dict, average_f1


def merge_config(args_config):
    config_file_path = args_config.config_path
    config_dict = yaml.safe_load(open(config_file_path))
    config = Config(config_dict[args_config.config_name])
    config.update_args(args_config)
    LOGGING.info("=" * 10 + "... Model Configs ..." + "=" * 10)
    LOGGING.info("{}".format(config.to_json_string()))
    bert_config = Config.from_json_file(os.path.join(args_config.bert_model, "config.json"))
    LOGGING.info("=" * 10 + "... PRETRAINED Model Configs ..." + "=" * 10)
    LOGGING.info("{}".format(bert_config.to_json_string()))
    return config


def main():
    args_config = args_parser()
    config = merge_config(args_config)

    if config.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    train_dataloader, dev_dataloader, test_dataloader = load_data(config, data_sign="conll")
    model, optimizer, sheduler, device, n_gpu = load_model(config)
    train(model, optimizer, sheduler, train_dataloader, dev_dataloader, test_dataloader, config, device, n_gpu)


if __name__ == "__main__":
    main()
