
export CUDA_VISIBLE_DEVICES=0

DATASET_NAME="rdrp_40_extend"
DATASET_TYPE="protein"
TASK_TYPE="binary_class"
MODEL_TYPE="sequence"
CONFIG_NAME="sequence_config.json"
INPUT_MODE="single"
LABEL_TYPE="rdrp"
SEQ_MAX_LENGTH=2048
TRUNCT_TYPE="right"
# # none, max, value_attention
SEQ_POOLING_TYPE="value_attention"
VOCAB_NAME="vocab.txt"
MAX_METRIC_TYPE="f1"
time_str=$(date "+%Y%m%d%H%M%S")
cd ..
python run.py \
  --data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE \
  --separate_file \
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
  --seq_vocab_path  ../vocab/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$VOCAB_NAME\
  --seq_pooling_type $SEQ_POOLING_TYPE \
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
  --trunc_type=$TRUNCT_TYPE \
  --no_token_type_embeddings
