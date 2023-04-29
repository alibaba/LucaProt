export CUDA_VISIBLE_DEVICES=0

cd ..

python predict_many_samples.py \
	--fasta_file ../data/rdrp/test/test.fasta  \
	--save_file ../result/rdrp/test_result.csv  \
	--emb_dir ../emb/   \
	--truncation_seq_length 4096  \
	--dataset_name rdrp_40_extend  \
	--dataset_type protein     \
	--task_type binary_class     \
	--model_type sefn     \
	--time_str 20230201140320   \
	--step 100000  \
	--threshold 0.5 \
	--print_per_number 10