#!/bin/sh

for model_name in Llama-2-7b-chat-hf  
do
	CUDA_VISIBLE_DEVICES=3 python3 run_unsupervised.py --multirun \
	method=semantic_space \
	preprocess=coqa \
	model_name=$model_name 
done


python .notify/notify.py --message="Your topo entropy script has finished" --token_id="8100080414:AAFZy6Tz0_bGSQXtw3m0HmlB3350zLMMYrA" --chat_id="226762806"