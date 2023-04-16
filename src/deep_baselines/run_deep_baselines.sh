#!/bin/bash

export CUDA_VISIBLE_DEVICES=$2

DATASET_NAME=$3
DATASET_TYPE="protein"
TASK_TYPE="binary_class"
MODEL_TYPE=$1
LABEL_TYPE="rdrp"
SEQ_MAX_LENGTH=2048
TRUNCT_TYPE="right"
VOCAB_NAME="deep_baselines_vocab.txt"
MAX_METRIC_TYPE="f1"
time_str=$(date "+%Y%m%d%H%M%S")

if [ $1 = "CHEER-CatWCNN" ]
then
    CONFIG_NAME="cheer_config.json"
    python run.py \
      --data_dir ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
      --separate_file \
      --filename_pattern {}_with_pdb_emb.csv \
      --dataset_name $DATASET_NAME \
      --dataset_type $DATASET_TYPE \
      --task_type $TASK_TYPE \
      --model_type $MODEL_TYPE \
      --label_type $LABEL_TYPE \
      --label_filepath ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
      --output_dir ../../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --log_dir ../../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --tb_log_dir ../../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --config_path ../../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
      --seq_vocab_path  ../../vocab/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$VOCAB_NAME\
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
      --seq_max_length=$SEQ_MAX_LENGTH \
      --embedding_trainable \
      --embedding_dim 128 \
      --channel_in 6 \
      --padding_idx 0 \
      --trunc_type=$TRUNCT_TYPE
elif [ $1 = "CHEER-WDCNN" ]
then
    CONFIG_NAME="cheer_config.json"
    python run.py \
      --data_dir ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
      --separate_file \
      --filename_pattern {}_with_pdb_emb.csv \
      --dataset_name $DATASET_NAME \
      --dataset_type $DATASET_TYPE \
      --task_type $TASK_TYPE \
      --model_type $MODEL_TYPE \
      --label_type $LABEL_TYPE \
      --label_filepath ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
      --output_dir ../../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --log_dir ../../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --tb_log_dir ../../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --config_path ../../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
      --seq_vocab_path  ../../vocab/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$VOCAB_NAME\
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
      --seq_max_length=$SEQ_MAX_LENGTH \
      --embedding_trainable \
      --embedding_dim 128 \
      --channel_in 6 \
      --padding_idx 0 \
      --trunc_type=$TRUNCT_TYPE
elif [ $1 = "CHEER-WCNN" ]
then
    CONFIG_NAME="cheer_config.json"
    python run.py \
      --data_dir ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
      --separate_file \
      --filename_pattern {}_with_pdb_emb.csv \
      --dataset_name $DATASET_NAME \
      --dataset_type $DATASET_TYPE \
      --task_type $TASK_TYPE \
      --model_type $MODEL_TYPE \
      --label_type $LABEL_TYPE \
      --label_filepath ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
      --output_dir ../../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --log_dir ../../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --tb_log_dir ../../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --config_path ../../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
      --seq_vocab_path  ../../vocab/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$VOCAB_NAME\
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
      --seq_max_length=$SEQ_MAX_LENGTH \
      --embedding_trainable \
      --embedding_dim 128 \
      --channel_in 6 \
      --padding_idx 0 \
      --trunc_type=$TRUNCT_TYPE
elif [ $1 = "VirHunter" ]
then
    CONFIG_NAME="virhunter_config.json"
    python run.py \
      --data_dir ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
      --separate_file \
      --filename_pattern {}_with_pdb_emb.csv \
      --dataset_name $DATASET_NAME \
      --dataset_type $DATASET_TYPE \
      --task_type $TASK_TYPE \
      --model_type $MODEL_TYPE \
      --label_type $LABEL_TYPE \
      --label_filepath ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
      --output_dir ../../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --log_dir ../../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --tb_log_dir ../../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --config_path ../../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
      --seq_vocab_path  ../../vocab/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$VOCAB_NAME\
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
      --seq_max_length=$SEQ_MAX_LENGTH \
      --embedding_trainable \
      --embedding_dim 128 \
      --padding_idx 0 \
      --trunc_type=$TRUNCT_TYPE
elif [ $1 = "Virtifier" ]
then
    CONFIG_NAME="virtifier_config.json"
    python run.py \
      --data_dir ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
      --separate_file \
      --filename_pattern {}_with_pdb_emb.csv \
      --dataset_name $DATASET_NAME \
      --dataset_type $DATASET_TYPE \
      --task_type $TASK_TYPE \
      --model_type $MODEL_TYPE \
      --label_type $LABEL_TYPE \
      --label_filepath ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
      --output_dir ../../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --log_dir ../../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --tb_log_dir ../../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --config_path ../../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
      --seq_vocab_path  ../../vocab/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$VOCAB_NAME\
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
      --seq_max_length=$SEQ_MAX_LENGTH \
      --embedding_trainable \
      --embedding_dim 128 \
      --bidirectional \
      --num_layers 2 \
      --hidden_dim 128 \
      --padding_idx 0 \
      --trunc_type=$TRUNCT_TYPE
elif [ $1 = "VirSeeker" ]
then
    CONFIG_NAME="virseeker_config.json"
    python run.py \
      --data_dir ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
      --separate_file \
      --filename_pattern {}_with_pdb_emb.csv \
      --dataset_name $DATASET_NAME \
      --dataset_type $DATASET_TYPE \
      --task_type $TASK_TYPE \
      --model_type $MODEL_TYPE \
      --label_type $LABEL_TYPE \
      --label_filepath ../../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
      --output_dir ../../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --log_dir ../../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --tb_log_dir ../../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$time_str \
      --config_path ../../config/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$CONFIG_NAME \
      --seq_vocab_path  ../../vocab/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$VOCAB_NAME\
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
      --seq_max_length=$SEQ_MAX_LENGTH \
      --embedding \
      --embedding_trainable \
      --bidirectional \
      --num_layers 2 \
      --hidden_dim 256 \
      --padding_idx 0 \
      --trunc_type=$TRUNCT_TYPE \
      --bias
else
    echo "Error, please input CHEER-CatWCNN, CHEER-WDCNN, CHEER-WCNN, VirHunter, Virtifier, Virseeker"
fi