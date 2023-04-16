#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATASET_NAME="rdrp_40_extend"
DATASET_TYPE="protein"
TASK_TYPE="binary_class"
LABEL_TYPE="rdrp"

MODEL_TYPE="xgb"
CONFIG_NAME="xgb_config.json"
time_str=$(date "+%Y%m%d%H%M%S")
python xgb.py \
  --data_dir ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
  --separate_file \
  --filename_pattern {}_with_pdb_emb.csv \
  --dataset_name $DATASET_NAME \
  --dataset_type $DATASET_TYPE \
  --task_type $TASK_TYPE \
  --label_type $LABEL_TYPE \
  --label_filepath ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt \
  --output_dir ../../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
  --log_dir ../../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
  --tb_log_dir ../../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
  --config_path ../../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --num_train_epochs=50 \
  --logging_steps=1000 \
  --save_steps=1000 \
  --overwrite_output_dir \
  --seed 1234 \
  --max_metric_type f1 \
  --early_stopping_rounds -1 \
  --grid_search \
  --pos_weight 40