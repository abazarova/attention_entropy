#!/bin/sh

# for model_name in Llama-2-13b-chat-hf
# do
# 	for dataset in ragtruth_qa ragtruth_summ
# 	do
# 		CUDA_VISIBLE_DEVICES=1 python3 run_condent.py --multirun \
# 		method=condent_rauq \
# 		method.cache_dir=cache/condent_v0/$dataset \
# 		transfer_names=[] \
# 		preprocess.val_size=100 \
# 		transfer.val_size=100 \
# 		model_name=$model_name \
# 		method.device=cuda \
# 		method.dtype=float16 \
# 		preprocess=$dataset \
# 		method.n_max=10 \
# 		evaluation.seed=42 \
# 		method.n_layers=40 \
# 		method.n_heads=40 \
# 		log_output=True \
# 		evaluation.val_size=100 \
# 		method.alpha=0.3,0.4,0.5,0.6 \
# 		method.aggregation=mean
# 	done
# done


# for model_name in Qwen2.5-7B-Instruct
# do
# 	CUDA_VISIBLE_DEVICES=1 python3 run_condent.py --multirun \
# 	method=condent_rauq \
# 	method.cache_dir=cache/condent_v0/squad \
# 	transfer_names=[] \
# 	preprocess.val_size=100 \
# 	transfer.val_size=100 \
# 	model_name=$model_name \
# 	method.device=cuda \
# 	method.dtype=float16 \
# 	preprocess=squad \
# 	method.n_max=10 \
# 	evaluation.seed=42 \
# 	method.n_layers=40 \
# 	method.n_heads=40 \
# 	log_output=True \
# 	evaluation.val_size=100 \
# 	method.alpha=0.3,0.4,0.5,0.6 \
# 	method.aggregation=mean \
# 	preprocess.source_dir=data/raw/SQuAD
# done


for model_name in Qwen2.5-7B-Instruct
do
	CUDA_VISIBLE_DEVICES=4 python3 run_condent.py --multirun \
	method=condent_rauq \
	method.cache_dir=cache/condent_v0/coqa \
	transfer_names=[] \
	preprocess.val_size=100 \
	transfer.val_size=100 \
	model_name=$model_name \
	method.device=cuda \
	method.dtype=bfloat16 \
	preprocess=coqa \
	method.n_max=10 \
	evaluation.seed=42 \
	method.n_layers=28 \
	method.n_heads=28 \
	log_output=True \
	evaluation.val_size=100 \
	method.alpha=0.3 \
	method.aggregation=mean
done


# for model_name in Qwen2.5-7B-Instruct
# do
# 	CUDA_VISIBLE_DEVICES=4 python3 run_condent.py --multirun \
# 	method=condent_rauq \
# 	method.cache_dir=cache/condent_v0/xsum \
# 	transfer_names=[] \
# 	preprocess.val_size=100 \
# 	transfer.val_size=100 \
# 	model_name=$model_name \
# 	method.device=cuda \
# 	method.dtype=bfloat16 \
# 	preprocess=xsum \
# 	method.n_max=10 \
# 	evaluation.seed=42 \
# 	method.n_layers=28 \
# 	method.n_heads=28 \
# 	log_output=True \
# 	evaluation.val_size=100 \
# 	method.alpha=0.3,0.4,0.5,0.6 \
# 	method.aggregation=mean
# done



python .notify/notify.py --message="Condent-RAUQ script has finished" --token_id="8100080414:AAFZy6Tz0_bGSQXtw3m0HmlB3350zLMMYrA" --chat_id="226762806"