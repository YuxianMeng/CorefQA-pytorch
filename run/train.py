#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 


import os
import yaml 
import random 
import argparse  
import numpy as np
import torch 
from torch import nn  

from model.corefqa import CorefQA



def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters 
    parser.add_argument("--config_path", default="/home/lixiaoya/", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default=None, type=str,)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--export_model", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/output")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--data_cache", type=bool, default=False)

    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
 
    return args 


def load_data(config):
    pass 


def load_model(config, num_train_epochs, label_list):
    device = torch.device("cuda")
    n_gpu = config.n_gpu 
    model = CorefQA(config)
    model.to(device)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    sheduler = None

    return model, optimizer, sheduler, device, n_gpu


def train(model, optimizer, sheduler,  train_dataloader, dev_dataloader, test_dataloader, config, \
    device, n_gpu,):
    
    pass 


def eval_checkpoint(model_object, eval_dataloader, config, \
    device, n_gpu, eval_sign="dev"):
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
    pass 


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, = load_model(config, num_train_steps)
    train(model, optimizer, sheduler, train_loader, dev_loader, \
        test_loader, config, device, n_gpu) 





if __name__ == "__main__":
    main()

















