#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


import os
import yaml 
import time 
import random 
import logging 
import argparse  
import numpy as np
import torch 

from config.load_config import Config
from data_loader.conll_dataloader import CoNLLDataLoader 
from model.corefqa import CorefQA
from module.optimization import AdamW, warmup_linear
from transformers.modeling import BertConfig


try:
    import torch_xla 
    import torch_xla.core.xla_model as xm 
except:
    print("=*="*10)
    print("IMPORT torch_xla when running on the TPU machine. ")
    print("=*="*10)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters 
    parser.add_argument("--config_path", default="/home/lixiaoya/bert.yaml", type=str)
    parser.add_argument("--config_name", default="spanbert_base", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default=None, type=str,)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--do_eval", default=bool, type=bool)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--eval_per_epoch", default=10, type=int) 
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--export_model", type=bool, default=False)
    parser.add_argument("--fp16_opt_level", type=str,default="O3",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",) 
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/output")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--data_cache", type=bool, default=False,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n") 
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--loss_scale", type=float, default=1.0, )
    parser.add_argument('--tpu', action='store_true', help="Whether to use tpu machine")
    parser.add_argument('--debug', action='store_true', help="print some debug information.")

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

    train_dataloader = dataloader.get_dataloader("train")
    dev_dataloader = dataloader.get_dataloader("dev")
    test_dataloader = dataloader.get_dataloader("test")

    return train_dataloader, dev_dataloader, test_dataloader, 


def load_model(config):

    if config.tpu:
        device = xm.xla_device()
        n_gpu = 0
    else:
        device = torch.device("cuda")
        print("-*-"*10)
        print("please notice that the device is :")
        print(device)
        print("-*-"*10)
        n_gpu = config.n_gpu 
    bert_config = BertConfig.from_json_file(os.path.join(config.bert_model, "config.json"))
    model = CorefQA(bert_config, config, device)
    
    model.to(device)

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
    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True
        )

    sheduler = None
    return model, optimizer, sheduler, device, n_gpu


def train(model, optimizer, sheduler,  train_dataloader, dev_dataloader, test_dataloader, config, \
    device, n_gpu,):

    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps = 0
    global_step = 0
    start_time = time.time()
    num_train_optimization_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    train_batches = [batch for batch in train_dataloader]
    eval_step = max(1, len(train_batches) // config.eval_per_epoch)


    for epoch in range(int(config.num_train_epochs)):
        print("=*="*20)
        print("start {} Epoch ... ".format(str(epoch)))
        model.train()
        logger.info("Start epoch #{} (lr = {})...".format(epoch, config.lr))

        if config.debug:
            print("INFO: start train the CorefQA Model.")
        
        for step, batch in enumerate(train_dataloader):
            ##if n_gpu == 1:
            ##    batch = tuple(t.to(device) for t in batch)
            doc_idx, sentence_map,subtoken_map, input_ids, input_mask, gold_mention_span, token_type_ids, attention_mask, \
                span_starts, span_ends, cluster_ids = batch["doc_idx"].squeeze(0), batch["sentence_map"].squeeze(0), None, \
                batch["flattened_input_ids"].view(-1, config.sliding_window_size), batch["flattened_input_mask"].view(-1, config.sliding_window_size), \
                batch["mention_span"].squeeze(0), None, None, batch["span_start"].squeeze(0), batch["span_end"].squeeze(0), batch["cluster_ids"].squeeze(0)
            doc_idx= doc_idx.to(device)
            sentence_map= sentence_map.to(device)
            # subtoken_map = subtoken_map.to(device)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            gold_mention_span = gold_mention_span.to(device)
            span_starts = span_starts.to(device)
            span_ends = span_ends.to(device)
            cluster_ids = cluster_ids.to(device)

            if config.debug and step % 2 == 0:
                print("INFO: The {} epoch training process {}.".format(epoch, step))

            loss = model(doc_idx=doc_idx, sentence_map=sentence_map, subtoken_map=subtoken_map, input_ids=input_ids, input_mask=input_mask, \
                gold_mention_span=gold_mention_span, token_type_ids=token_type_ids, attention_mask=attention_mask, span_starts=span_starts, span_ends=span_ends, cluster_ids=cluster_ids)
            print("loss")
            if n_gpu > 1:
                loss = loss.mean()
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if config.tpu:
                xm.optimizer_step(optimizer, barrier=True)

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.fp16:
                    lr_this_step = config.lr * warmup_linear(global_step/ num_train_optimization_steps, config.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % eval_step == 0:
                logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                    epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))

                save_model = False
                if config.do_eval:
                    result, _, _ = evaluate(config, model, device, dev_dataloader, test_dataloader, n_gpu)
                model.train()
                result['global_step'] = global_step
                result['epoch'] = epoch
                result['learning_rate'] = config.lr
                result['batch_size'] = config.train_batch_size

                if save_model:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(config.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(config.output_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(config.output_dir)
                    if best_result:
                        with open(os.path.join(config.output_dir, "eval_results.txt"), "w") as writer:
                            for key in sorted(best_result.keys()):
                                writer.write("%s = %s\n" % (key, str(best_result[key])))



def evaluate(config, model, device, dev_dataloader, test_dataloader, n_gpu, eval_sign="dev"):
    model_object.eval()

    eval_loss = 0 
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    end_gold_lst = []
    span_gold_lst = []

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1  


def merge_config(args_config):
    config_file_path = args_config.config_path 
    config_dict = yaml.safe_load(open(config_file_path))
    config = Config(config_dict[args_config.config_name])
    config.update_args(args_config)
    # config.print_config()
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







