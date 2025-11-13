import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Qwen2Tokenizer,
)
from transformers.tokenization_utils_base import BatchEncoding

load_dotenv()


def load_model_and_tokenizer(model_id: str, device: torch.device) -> tuple:
    """Load the model and the corresponding tokenizer and place the model at the selected device."""
    login(os.environ["HUGGING_FACE_API_KEY"])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = "bfloat16" if "Qwen" in model_id else "float16"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_dataset(path: Path):
    """Load dataset from the given path."""
    with open(path) as f:
        data = json.load(f)
    return data


def generate_prompt_ids(
    context: str,
    model_name: str,
    tokenizer: LlamaTokenizer | LlamaTokenizerFast | Qwen2Tokenizer,
) -> BatchEncoding:
    """Generate the prompt for the model given the dataset."""
    text = f"Summarize the article in one sentence.\n\nArticle: {context}\n Summary:"
    if model_name in [
        "Mistral-7B-Instruct-v0.1",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
    ]:
        text = f"[INST] {text} [/INST]"
        prompt_ids = tokenizer(text, return_tensors="pt")
    elif model_name == "Llama-3.1-8B-Instruct":
        messages = [
            {"role": "user", "content": text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert type(text) is str, "Incorrect output of tokenizer.apply_chat_template"
        prompt_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")

    elif model_name == "Qwen3-8B":
        messages = [
            {"role": "user", "content": text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert type(text) is str, "Incorrect output of tokenizer.apply_chat_template"
        prompt_ids = tokenizer(text, return_tensors="pt")
    else:
        raise NotImplementedError
    return prompt_ids


def save_generations(generations: list, save_path: Path):
    """Save generations of the model."""
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "generations.json", "w") as f:
        json.dump(generations, f)

    df = pd.DataFrame(generations)
    df.to_csv(save_path / "generations.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("dataset", type=str, choices=["XSum"])
    parser.add_argument("--data_dir", type=str, default="/app/data/fixed/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    model_id = args.model_id
    model_name = model_id.split("/")[1]
    dataset = args.dataset
    data_dir = args.data_dir
    device = torch.device(args.device)
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens

    root_dir = Path(args.data_dir) / args.dataset
    load_path = root_dir / "preprocessed.json"
    save_path = root_dir / args.model_id.split("/")[1]

    if not load_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {load_path}")
    data = load_dataset(load_path)

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, device=device)

    generations = []
    for sample in tqdm(data, desc=f"Generating answers for {dataset}"):
        context = sample["document"].strip()
        input_ids = generate_prompt_ids(context, model_name, tokenizer).to(device)
        sample["prompt"] = tokenizer.decode(input_ids.input_ids[0])
        generated_ids = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0][input_ids.input_ids.shape[1] :].cpu()
        summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
        sample["generated_summary"] = summary
        generations.append(sample)

    save_generations(generations, save_path)
