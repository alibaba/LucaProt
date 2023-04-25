export CUDA_VISIBLE_DEVICES=0
cd ..
python predict.py --data_path  ../data/rdrp/demo/demo.csv \
    --emb_dir ../data/rdrp/demo/embs/esm2_t36_3B_UR50D \
    --dataset_name rdrp_40_extend --dataset_type protein --task_type binary_class --model_type sefn \
    --time_str 20230201140320 --step 100000 --evaluate --threshold 0.5 --batch_size 16 --print_per_batch 1
