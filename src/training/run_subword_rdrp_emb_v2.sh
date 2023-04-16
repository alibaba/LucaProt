#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATASET_NAME="rdrp_40_extend"
DATASET_TYPE="protein"
TASK_TYPE="binary_class"
MODEL_TYPE="embedding"
CONFIG_NAME="embedding_config.json"
INPUT_MODE="single"
LABEL_TYPE="rdrp"
embedding_input_size=2560
# matrix need pooling(value_attention) bos unneed(bos none)
embedding_type="matrix"
# max, value_attention
embedding_pooling_type="value_attention"
embedding_max_length=2048
TRUNCT_TYPE="right"
MAX_METRIC_TYPE="f1"
time_str=$(date "+%Y%m%d%H%M%S")
cd ..
python run.py \
  --data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
  --tfrecords \
  --filename_pattern {}_with_pdb_emb.csv \
  --dataset_name $DATASET_NAME \
  --dataset_type $DATASET_TYPE \
  --task_type $TASK_TYPE \
  --model_type $MODEL_TYPE \
  --input_mode $INPUT_MODE \
  --label_type $LABEL_TYPE \
  --label_filepath ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
  --output_dir ../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
  --log_dir ../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
  --tb_log_dir ../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
  --config_path ../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --num_train_epochs=50 \
  --logging_steps=1000 \
  --save_steps=1000 \
  --overwrite_output_dir \
  --sigmoid \
  --loss_type bce \
  --max_metric_type $MAX_METRIC_TYPE \
  --trunc_type=$TRUNCT_TYPE \
  --embedding_pooling_type $embedding_pooling_type \
  --embedding_max_length=$embedding_max_length \
  --embedding_input_size $embedding_input_size\
  --embedding_type $embedding_type \
  --shuffle_queue_size 10000
