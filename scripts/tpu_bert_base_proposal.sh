#!/usr/bin/env python3
# -*- coding: utf-8 -*-

export PYTHONPATH="$PYTHONPATH:/mnt/data/CorefQA-pytorch"
export TPU_IP_ADDRESS=10.173.250.154
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

echo ${TPU_IP_ADDRESS}
echo ${XRT_TPU_CONFIG}


EXP_ID=26_22_23
LOG_FILE=/mnt/data/CorefQA-pytorch/${EXP_ID}.txt
FOLDER_PATH=/mnt/data/CorefQA-pytorch
CONFIG_PATH=${FOLDER_PATH}/config/gpu_spanbert.yml
DATA_PATH=/mnt/data/tpu-data/data
BERT_PATH=/mnt/data/ckpt/spanbert_base
output_path=/mnt/data/ckpt/spanbert_base


config_name=spanbert_base
learning_rate=3e-5
num_train_epoch=20
eval_per_epoch=3
warmup_proportion=-1
seed=2333
n_gpu=1
mention_chunk_size=200000


nohup python3 ${FOLDER_PATH}/run/train.py \
--n_gpu ${n_gpu} \
--mention_chunk_size ${mention_chunk_size} \
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
--tpu \
--mention_proposal_only \
--use_cache_data \
--do_eval true \
--save_model \
> $LOG_FILE 2>&1 & tail -f $LOG_FILE
