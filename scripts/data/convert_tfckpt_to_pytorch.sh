#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 

model_sign=$1


if [[ ${model_sign} = "base" ]]; then
    model_name=cased_L-12_H-768_A-12
elif [[ ${model_sign} = "large" ]]; then
    model_name=cased_L-24_H-1024_A-16
fi 


export BERT_BASE_DIR=/dev/shm/xiaoya/pretrain_ckpt/${model_name}


transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin