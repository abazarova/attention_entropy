import argparse
import json
import os
from pathlib import Path

import nltk
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
    StoppingCriteriaList,
)
from transformers.tokenization_utils_base import BatchEncoding
from utils import STOP_SEQUENCES, StoppingCriteriaSub

load_dotenv()


def load_model_and_tokenizer(model_id: str, device: str) -> tuple:
    """Load the model and the corresponding tokenizer and place the model at the selected device."""
    login(os.environ["HUGGING_FACE_API_KEY"])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = "bfloat16" if "Qwen" in model_id else "float16"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_dataset(dataset: str, path: Path):
    """Load dataset from the given path."""
    if dataset == "CoQA":
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
    return data


def generate_prompt_ids(
    question: str,
    context: str,
    dataset: str,
    model_name: str,
    tokenizer: Qwen2Tokenizer | LlamaTokenizer | LlamaTokenizerFast,
) -> BatchEncoding:
    """Generate the prompt for the model given the dataset."""
    if dataset == "CoQA":
        text = context + " Q: " + question + " A:"
    elif dataset == "HotpotQA":
        text = f"Using the following knowledge: [{context}], answer the question in a brief but complete sentence: {question} "
    else:
        text = (
            "Given the context, answer the question in a single brief but complete sentence. "
            + "Note that your answer should be strictly based on the given context. "
            + "In case the context does not contain the necessary information to answer the question, "
            + 'please reply with: "Unable to answer based on given context."\n'
            + f"Context: {context}\n"
            + f"Question: {question}\n"
            + "Answer: "
        )

    if model_name in [
        "Mistral-7B-Instruct-v0.1",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
    ]:
        if dataset != "CoQA":
            text = f"[INST] {text} [/INST]"
        prompt_ids = tokenizer(text, return_tensors="pt")
    elif model_name == "Llama-3.1-8B-Instruct":
        if dataset != "CoQA":
            messages = [
                {"role": "user", "content": text},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            assert type(text) is str, (
                "Incorrect output of tokenizer.apply_chat_template"
            )
        prompt_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    elif model_name == "Qwen3-8B":
        if dataset == "CoQA":
            text = f"Given the context and the subsequent series of question-answer pairs, answer the last question.\n\n{text}"
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


def generate_answer(
    model,
    input_ids: BatchEncoding,
    dataset: str,
    tokenizer: LlamaTokenizer | Qwen2Tokenizer | LlamaTokenizerFast,
    max_new_tokens: int,
    temperature: float,
    stop_sequences: str | None,
):
    """Generate answers to the given query."""
    stopping_criteria = None
    if stop_sequences == "default" and dataset == "CoQA":
        stopping_criteria = StoppingCriteriaList(
            [
                StoppingCriteriaSub(
                    stops=STOP_SEQUENCES,
                    initial_length=input_ids.input_ids.shape[1],
                    tokenizer=tokenizer,
                )
            ]
        )

    generated_ids = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )[0][input_ids.input_ids.shape[1] :].cpu()

    if dataset == "CoQA":
        generation = postprocess_generation(generated_ids, tokenizer)
    else:
        generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generation


def postprocess_generation(
    generated_ids: torch.Tensor,
    tokenizer: LlamaTokenizer | Qwen2Tokenizer | LlamaTokenizerFast,
) -> str:
    """Decode the generation and postprocess it."""
    # Decode the generation
    generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Find the earliest occurrence of any stop sequence
    earliest_stop_pos = len(generation)
    for stop in STOP_SEQUENCES:
        stop_pos = generation.find(stop)
        if stop_pos != -1 and stop_pos < earliest_stop_pos:
            # Special handling for "A:" - don't remove it if it's at position 0
            if stop in ["A:", "Answer:"] and stop_pos == 0:
                continue  # Skip "A:" at the beginning
            earliest_stop_pos = stop_pos

    # Truncate at the earliest stop sequence
    if earliest_stop_pos < len(generation):
        sliced_answer = generation[:earliest_stop_pos]
    else:
        sliced_answer = generation

    return sliced_answer


def save_generations(generations: list, save_path: Path):
    """Save generations of the model."""
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "generations.json", "w") as f:
        json.dump(generations, f)

    df = pd.DataFrame(generations)
    df.to_csv(save_path / "generations.csv", index=False)


if __name__ == "__main__":
    nltk.download("wordnet")
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.PorterStemmer()

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument(
        "dataset",
        type=str,
        choices=["SQuAD", "CoQA", "HotpotQA", "NQ_Swap"],
    )
    parser.add_argument("--data_dir", type=str, default="/app/data/fixed/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--stop_sequences", type=str, default="default")

    args = parser.parse_args()

    model_id = args.model_id
    model_name = model_id.split("/")[1]
    dataset = args.dataset
    data_dir = args.data_dir
    device = args.device
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    stop_sequences = args.stop_sequences

    if "Qwen" in model_name:
        stop_sequences = None

    root_dir = Path(data_dir) / dataset
    load_path = root_dir / "preprocessed.json"
    save_path = root_dir / model_name

    if not load_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {load_path}")
    data = load_dataset(dataset, load_path)

    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)

    generations = []
    for qa in tqdm(data, desc=f"Generating answers for {dataset}"):
        if not qa["answers"]:
            continue
        question = qa["question"].strip()
        context = qa["context"].strip()

        input_ids = generate_prompt_ids(
            question,
            context,
            dataset=dataset,
            model_name=model_name,
            tokenizer=tokenizer,
        ).to(device)
        qa["prompt"] = tokenizer.decode(input_ids.input_ids[0])
        answer = generate_answer(
            model=model,
            input_ids=input_ids,
            dataset=dataset,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

        qa["generated_answer"] = answer
        generations.append(qa)

    save_generations(generations, save_path)
