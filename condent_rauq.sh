#!/bin/sh

for model_name in Llama-2-7b-chat-hf
	do
	for dataset in coqa squad 
	do
		for aggregation in mean
		do
			for alpha in 0.5
			do
				CUDA_VISIBLE_DEVICES=3 python3 run_condent.py --multirun \
				method=condent_rauq \
				method.cache_dir=cache/condent_v0/$dataset \
				transfer_names=[] \
				preprocess.val_size=100 \
				transfer.val_size=100 \
				model_name=$model_name \
				method.device=cuda \
				method.dtype=float16 \
				preprocess=$dataset \
				method.n_max=10 \
				evaluation.seed=42 \
				method.n_layers=32 \
				method.n_heads=32 \
				log_output=True \
				evaluation.val_size=100 \
				method.alpha=$alpha \
				method.aggregation=$aggregation
				#preprocess.source_dir=data/fixed/SQuAD
			done
		done
	done
done



python .notify/notify.py --message="Condent-RAUQ script has finished" --token_id="8100080414:AAFZy6Tz0_bGSQXtw3m0HmlB3350zLMMYrA" --chat_id="226762806"