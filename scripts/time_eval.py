import numpy as np
import pandas as pd
import torch

from src.methods.llm_base import LLMBase
from src.preprocess.PreprocessRAGTruth import RAGTruth

model_name = "Mistral-7B-Instruct-v0.1"

device = "cuda:4"

ragtruth_data_config = {
    "task_type": "QA",
    "source_dir": "data/raw/RAGTruth",
    "save_dir": "data/processed/RAGTruth",
    "split": "original",  # train_test_split, no_single_test
    "test_size": 0.25,
    "random_state": 42,
}

generation_config = {
    "dtype": "float16",
    "num_return_sequences": 1,
    "temperature": 1.0,
    "max_new_tokens": 512,
}

dataset = RAGTruth(
    model_name=model_name,
    **ragtruth_data_config,
)


X, y, train_indices, test_indices = dataset.process()
X, y = X.iloc[:16], y[:16]
X["name"] = ["ragtruth_qa"] * 16

Xllm = LLMBase(
    model_name=model_name,
    device=device,
    **generation_config,
)

WARMUP = 3
ITERS = 10


def inference_cuda_generate(model, inputs, warmup, iterations):
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    timings = np.zeros(iterations)

    with torch.no_grad():
        # GPU-WARM-UP
        for _ in range(warmup):
            # print(model)
            _ = model.generate_llm_responses(
                X=inputs,
            )
        torch.cuda.synchronize()
        # MEASURE PERFORMANCE
        for rep in range(iterations):
            starter.record()
            _ = list(
                model.generate_llm_responses(
                    X=inputs,
                )
            )
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    return timings


time_array = inference_cuda_generate(
    model=Xllm,
    inputs=X,
    warmup=WARMUP,
    iterations=ITERS,
)

print(f"Generation: {time_array.mean():.3f} ± {time_array.std():.3f} ms")

ans_df = dict()
ans_df["Generation"] = time_array.tolist()

ans_df = pd.DataFrame(ans_df)
ans_df.to_csv("inference_times.csv")

print("Done!")
print("Done!")
