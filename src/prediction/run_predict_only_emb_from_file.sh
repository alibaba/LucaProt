#!/bin/bash

# sh run_predict_seq_emb_from_file.sh rdrp_40_extend embedding 20230201140320 100000 64 1000
if [ $# -lt 5 ]
then
  echo "args num < 5"
fi
dataset_name=$1
model_type=$2
time_str=$3
step=$4
batch_size=$5
emb_dir=$6

CUDA_VISIBLE_DEVICES=0
if [ $# -ge 7 ]
then
  CUDA_VISIBLE_DEVICES=$7
fi

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

print_per_batch=1000
if [ $# -ge 8 ]
then
  print_per_batch=$8
fi

cd ..
python predict.py --data_path ../data/rdrp/2022Cell_RdRP_with_pdb_emb.csv \
    --dataset_name $dataset_name --dataset_type protein --task_type binary_class --model_type $model_type \
    --emb_dir $emb_dir/2022Cell_RdRP/esm2_t36_3B_UR50D \
    --time_str $time_str --step $step --evaluate --threshold 0.5 --batch_size $batch_size --print_per_batch $print_per_batch

python predict.py --data_path ../data/rdrp/2022Science_RdRP_with_pdb_emb.csv \
    --emb_dir $emb_dir/2022Science_RdRP/esm2_t36_3B_UR50D \
    --dataset_name $dataset_name --dataset_type protein --task_type binary_class --model_type $model_type \
    --time_str $time_str --step $step --evaluate --threshold 0.5 --batch_size $batch_size --print_per_batch $print_per_batch

python predict.py --data_path ../data/rdrp/2022NM_RdRP_with_pdb_emb.csv \
    --emb_dir $emb_dir/2022NM_RdRP/esm2_t36_3B_UR50D \
    --dataset_name $dataset_name --dataset_type protein --task_type binary_class --model_type $model_type \
    --time_str $time_str --step $step --evaluate --threshold 0.5 --batch_size $batch_size --print_per_batch $print_per_batch


python predict.py --data_path ../data/rdrp/All_RT_with_pdb_emb.csv \
    --emb_dir $emb_dir/All_RT/esm2_t36_3B_UR50D \
    --dataset_name $dataset_name --dataset_type protein --task_type binary_class --model_type $model_type \
    --time_str $time_str --step $step --evaluate --threshold 0.5 --batch_size $batch_size --print_per_batch $print_per_batch


python predict.py --data_path ../data/rdrp/Eukaryota_DdRP_with_pdb_emb.csv \
    --emb_dir $emb_dir/Eukaryota_DdRP/esm2_t36_3B_UR50D \
    --dataset_name $dataset_name --dataset_type protein --task_type binary_class --model_type $model_type \
    --time_str $time_str --step $step --evaluate --threshold 0.5 --batch_size $batch_size --print_per_batch $print_per_batch


python predict.py --data_path ../data/rdrp/Eukaryota_RdRP_with_pdb_emb.csv \
    --emb_dir $emb_dir/Eukaryota_RdRP/esm2_t36_3B_UR50D \
    --dataset_name $dataset_name --dataset_type protein --task_type binary_class --model_type $model_type \
    --time_str $time_str --step $step --evaluate --threshold 0.5 --batch_size $batch_size --print_per_batch $print_per_batch


python predict.py --data_path ../data/rdrp/ours_checked_rdrp_final.csv \
    --emb_dir $emb_dir/../../all_predicted_checked_unchecked/emb/esm2_t36_3B_UR50D \
    --dataset_name $dataset_name --dataset_type protein --task_type binary_class --model_type $model_type \
    --time_str $time_str --step $step --evaluate --threshold 0.5 --batch_size $batch_size --print_per_batch $print_per_batch


cd ./deep_baselines/
python statistics.py --dataset_name $dataset_name --model_type $model_type --time_str $time_str --step $step

