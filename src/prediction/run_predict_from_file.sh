export CUDA_VISIBLE_DEVICES=0
cd ..
python predict.py --data_path ../data/all_500aa.part_001_001_with_pdb_emb.csv \
    --emb_dir ../data/all/part_001_001/esm2_t36_3B_UR50D \
    --dataset_name rdrp_40_extend --dataset_type protein --task_type binary_class --model_type sefn \
    --time_str 20230201140320 --step 100000 --evaluate --threshold 0.5 --batch_size 128 --print_per_batch 1000
