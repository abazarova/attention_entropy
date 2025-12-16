#!/bin/sh


for model_name in Llama-2-7b-chat-hf #Mistral-7B-Instruct-v0.1 
do
	CUDA_VISIBLE_DEVICES=1 python3 run_unsupervised.py --multirun \
	method=topo_entropy \
	preprocess.val_size=100 \
	transfer.val_size=100 \
	model_name=$model_name \
	method.device=cuda \
	method.dtype=float32 \
	preprocess=coqa \
	transfer_names="[]" \
	evaluation.seed=42 \
	method.cache_dir=cache/generated_responses
done



# for model_name in Mistral-7B-Instruct-v0.1 #Llama-2-7b-chat-hf 
# do
# 	CUDA_VISIBLE_DEVICES=4 python3 run_unsupervised.py --multirun \
# 	method=topo_entropy \
# 	preprocess.val_size=100 \
# 	transfer.val_size=100 \
# 	model_name=$model_name \
# 	method.device=cuda \
# 	method.dtype=float32 \
# 	preprocess=squad \
# 	transfer_names="[]" \
# 	#preprocess.source_dir=/app/raw/SQuAD \
# 	evaluation.seed=42 \
# 	method.cache_dir=cache/generated_responses
# done

# for model_name in Mistral-7B-Instruct-v0.1 #Llama-2-7b-chat-hf 
# do
# 	CUDA_VISIBLE_DEVICES=4 python3 run_unsupervised.py --multirun \
# 	method=topo_entropy \
# 	preprocess.val_size=100 \
# 	transfer.val_size=100 \
# 	model_name=$model_name \
# 	method.device=cuda \
# 	method.dtype=float32 \
# 	preprocess=xsum \
# #	preprocess.source_dir=/app/raw/XSum \
# 	transfer_names="[]" \
# 	evaluation.seed=42 \
# 	method.cache_dir=cache/generated_responses
# done

python .notify/notify.py --message="Your topo entropy script has finished" --token_id="8100080414:AAFZy6Tz0_bGSQXtw3m0HmlB3350zLMMYrA" --chat_id="226762806"