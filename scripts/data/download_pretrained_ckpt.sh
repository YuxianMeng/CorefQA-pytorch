#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# Author: xiaoy li 
# description:
# download pretrained model ckpt 

BERT_PRETRAIN_CKPT=/home/lixiaoya/pretrain_ckpt

mkdir -p ${BERT_PRETRAIN_CKPT}
echo "DownLoad BERT Cased Base"
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip -P ${BERT_PRETRAIN_CKPT}
unzip ${BERT_PRETRAIN_CKPT}/cased_L-12_H-768_A-12.zip -d ${BERT_PRETRAIN_CKPT}
rm ${BERT_PRETRAIN_CKPT}/cased_L-12_H-768_A-12.zip


echo "DownLoad BERT Cased Large"
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip -P ${BERT_PRETRAIN_CKPT}
unzip ${BERT_PRETRAIN_CKPT}/cased_L-24_H-1024_A-16.zip -d ${BERT_PRETRAIN_CKPT}
rm ${BERT_PRETRAIN_CKPT}/cased_L-24_H-1024_A-16.zip


echo "DownLoad Span-BERT Cased Base"
wget https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz -P ${BERT_PRETRAIN_CKPT} 
tar -zxvf ${BERT_PRETRAIN_CKPT}/spanbert_hf_base.tar.gz -C ${BERT_PRETRAIN_CKPT}
rm ${BERT_PRETRAIN_CKPT}/spanbert_hf_base.tar.gz


echo "DownLoad Span-BERT Cased Large"
wget https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf.tar.gz -P ${BERT_PRETRAIN_CKPT}
tar -zxvf ${BERT_PRETRAIN_CKPT}/spanbert_hf.tar.gz -C ${BERT_PRETRAIN_CKPT}
rm ${BERT_PRETRAIN_CKPT}/spanbert_hf.tar.gz