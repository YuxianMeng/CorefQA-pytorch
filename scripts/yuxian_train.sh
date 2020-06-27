#!/usr/bin/env bash
# -*- coding: utf-8 -*-



exp_id=22_1
FOLDER_PATH=/home/lixiaoya/yuxian/coref
CONFIG_PATH=${FOLDER_PATH}/config/gpu_spanbert.yml
DATA_PATH=/dev/shm/xiaoya/data
#BERT_PATH=/dev/shm/xiaoya/pretrain_ckpt/cased_L-12_H-768_A-12
#BERT_PATH=/dev/shm/xiaoya/spanbert_finetune/finetune_spanbert_large_quoref
#BERT_PATH=/dev/shm/xiaoya/spanbert_finetune/finetune_spanbert_base_squad2
#BERT_PATH=/dev/shm/xiaoya/spanbert_finetune/finetune_spanbert_large_squad2
EXPORT_DIR=/dev/shm/xiaoya/yuxian/test_output


exp_id=2020.06.24_morn
config_name=spanbert_base
learning_rate=3e-5
num_train_epoch=200
eval_per_epoch=3
warmup_proportion=-1
seed=2333
n_gpu=1


output_path=/dev/shm/xiaoya/yuxian/corefqa_pytorch_output/${exp_id}


mkdir -p ${output_path}
export PYTHONPATH=${FOLDER_PATH}

export CUDA_VISIBLE_DEVICES=3
LOG_FILE="test_save_model.log"

python3 ${FOLDER_PATH}/run/train.py \
--n_gpu ${n_gpu} \
--config_path ${CONFIG_PATH} \
--config_name ${config_name} \
--data_dir ${DATA_PATH} \
--bert_model ${BERT_PATH} \
--lr ${learning_rate} \
--eval_per_epoch ${eval_per_epoch} \
--warmup_proportion ${warmup_proportion} \
--num_train_epochs ${num_train_epoch} \
--seed ${seed} \
--output_dir ${output_path} \
--mention_proposal_only \
--use_cache_data \
--do_eval true \
--save_model \
> $LOG_FILE 2>&1 & tail -f $LOG_FILE
