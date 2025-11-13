#!/bin/sh

for model_name in Llama-2-7b-chat-hf Llama-2-13b-chat-hf Llama-3.1-8B-Instruct Qwen3-8B Mistral-7B-Instruct-v0.1
do
	CUDA_VISIBLE_DEVICES=4 python3 run_unsupervised.py --multirun \
	method=selfcheckgpt_nli \
	preprocess.val_size=100 \
	transfer.val_size=100 \
	model_name=$model_name \
	method.device=cuda \
	method.dtype=float16 \
	preprocess=squad,xsum \
	transfer_names="[]" \
	evaluation.seed=42 
done

python .notify/notify.py --message="Your tokenwise entropy script has finished" --token_id="8100080414:AAFZy6Tz0_bGSQXtw3m0HmlB3350zLMMYrA" --chat_id="226762806"