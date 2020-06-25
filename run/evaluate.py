#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 




import os
import yaml 
import random 
import argparse  
import numpy as np
import torch 


import data_preprocess.conll as conll 
from config.load_config import Config
from transformers.modeling import BertConfig
from data_loader.conll_dataloader import CoNLLDataLoader 
from model.corefqa import CorefQA
from module import metrics 


try:
    import torch_xla 
    import torch_xla.core.xla_model as xm 
except:
    print("=*="*10)
    print("IMPORT torch_xla when running on the TPU machine. ")
    print("=*="*10)



def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters 
    parser.add_argument("--config_path", default="/home/lixiaoya/bert.yaml", type=str)
    parser.add_argument("--config_name", default="spanbert_base", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--data_cache", type=bool, default=False,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n") 
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--tpu', action='store_true', help="Whether to use tpu machine")
    parser.add_argument('--debug', action='store_true', help="print some debug information.")
    parser.add_argument('--eval_result_log', type=str, default="result.txt")
    parser.add_argument('--eval_ckpt_path', type=str, default="/home/lixiaoya/corefqa_output_ckpt")

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

    test_dataloader = dataloader.get_dataloader("test")

    return test_dataloader 


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
    eval_ckpt_path = torch.load(config.eval_ckpt_path) 
    model.load_state_dict(eval_ckpt_path)
    model.to(device)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, device, n_gpu

def evaluate(config, model_object, device, dataloader, n_gpu, eval_sign="test", official_stdout=False):
    
    model_object.eval()
    print("###"*20)
    print("="*8 + "Evaluate {} dataset".format(eval_sign) + "="*8)
    gold_cluster = []
    pred_cluster = []
    coref_prediction = {}
    coref_evaluator = metrics.CorefEvaluator()

    # top_span_starts, top_span_ends, predicted_antecedents, predicted_clusters
    for case_idx, case_feature in enumerate(dataloader):
        doc_idx, sentence_map, subtoken_map, input_ids, input_mask = case_feature["doc_idx"].squeeze(0), \
            case_feature["sentence_map"].squeeze(0), case_feature["flattened_input_ids"].view(-1, config.sliding_window_size), case_feature["flattened_input_mask"].view(-1, config.sliding_window_size)

        gold_cluster_ids = case_feature["cluster_ids"].squeeze(0)

        pred_cluster_ids = model(doc_idx=doc_idx, sentence_map=sentence_map, subtoken_map=subtoken_map, \
            input_ids=input_ids, input_mask=input_mask)

        coref_prediction[doc_idx] = pred_cluster_ids 
        coref_evaluator.update(pred_cluster_ids, gold_cluster_ids, mention_to_predict, mention_to_gold)

    summary_dict = {}
    conll_results = conll.evaluate_conll(config.eval_path, coref_prediction, official_stdout)
    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    summary_dict["Average F1 (conll)"] = average_f1 
    print("Average F1 (conll) : {:.2f}%".format(average_f1))

    p, r, f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f 
    print("Average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Average precision (py)"] = p 
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r 
    print("Average recall (py): {:.2f}%".format(r * 100))
    print("###"*20)


    with open(config.eval_result_log, "w") as writer:
        writer.write("Test RESULTS: \n")
        for key in sorted(summary_dict.keys()):
            writer.write("Test: %s = %s\n" % (key, str(summary_dict[key])))
        writer.write("TEST Average (conll) F1 : %s" % (str(average_f1)))

    return summary_dict, average_f1 


def merge_config(args_config):
    config_file_path = args_config.config_path 
    config_dict = yaml.safe_load(open(config_file_path))
    config = Config(config_dict[args_config.config_name])
    config.update_args(args_config)
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

    test_dataloader = load_data(config, data_sign="conll")
    model, device, n_gpu = load_model(config)
    summary_dict, average_f1 = evaluate(config, model, device, test_dataloader, n_gpu, eval_sign="test")




if __name__ == "__main__":
    main()

