#!/bin/bash

# Define the datasets and models
DATASETS=("SQuAD")
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.1"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
)
# Loop through each dataset and model combination
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "=============================================="
        echo "Running calc_rougel.py with:"
        echo "Dataset: $dataset"
        echo "Model: $model"
        echo "=============================================="
        
        # Run the script
        python scripts/calc_rougel.py \
            "$model" \
            "$dataset" \
        
        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed: $dataset with $model"
        else
            echo "❌ Failed: $dataset with $model"
            # Uncomment the next line if you want to stop on first error
            # exit 1
        fi
        
        echo ""
        sleep 1  # Small delay between runs
    done
done

echo "All jobs completed!"