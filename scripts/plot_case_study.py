import argparse
import json
import os
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from scipy.stats import entropy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

load_dotenv()

meta_dict = {
    "Mistral-7B-Instruct-v0.1": ("mistralai/Mistral-7B-Instruct-v0.1", 32),
    "Llama-2-7b-chat-hf": ("meta-llama/Llama-2-7b-chat-hf", 32),
    "Llama-2-13b-chat-hf": ("meta-llama/Llama-2-13b-chat-hf", 40),
    "Llama-3.1-8B-Instruct": ("meta-llama/Llama-3.1-8B-Instruct", 32),
    "Meta-Llama-3-8B-Instruct": ("meta-llama/Meta-Llama-3-8B-Instruct", 32),
    "Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct", 28),
}


def process_samples_sequentially(df, tokenizer, llm, answer_token, n_l, mtd_path, args):
    """Process samples sequentially but optimized"""
    device = torch.device("cuda")
    llm = llm.to(device)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        sample = row["entire_text"]
        input_ids = tokenizer(sample, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        answ_pos = (
            tokenizer(
                sample[: sample.rfind(answer_token)],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].shape[-1]
            + 1
        )

        with torch.no_grad():
            outputs = llm(input_ids, output_attentions=True)
            attention_maps = outputs.attentions

        for l in range(n_l):
            for h in range(n_l):
                attn_map = attention_maps[l][0][h].cpu()
                distance_mx = attn_map_to_dist_mx(attention_map=attn_map)
                df.at[i, f"mtd_{l}_{h}"] = load_mtd(row, l, h, mtd_path)
                if args.entropy:
                    df.at[i, f"entropy_{l}_{h}"] = exact_token_entropy(
                        attn_map, answ_pos
                    )
                if args.cluster_distance:
                    df.at[i, f"cluster_distance_{l}_{h}"] = distance_between_clusters(
                        distance_mx, answ_pos
                    )
                if args.n_edges:
                    n_edges = n_edges_between_clusters(distance_mx, answ_pos)
                    df.at[i, f"n_edges_{l}_{h}"] = n_edges

        # Clear memory after each sample
        del input_ids, outputs, attention_maps
        torch.cuda.empty_cache()

    return df


def load_mtd(row, i: int, j: int, root_dir: Path):
    sample_id = row["id"]
    if "Unnamed: 0" in row.axes[0]:
        sample_idx = row["Unnamed: 0"]
        if os.path.exists(root_dir / f"layer_{i}/head_{j}/{sample_idx}.json"):
            with open(root_dir / f"layer_{i}/head_{j}/{sample_idx}.json") as f:
                data = json.load(f)
    elif os.path.exists(root_dir / f"layer_{i}/head_{j}/{sample_id}.json"):
        with open(root_dir / f"layer_{i}/head_{j}/{sample_id}.json") as f:
            data = json.load(f)
    else:
        raise FileNotFoundError
    return data["mtopdiv"] / data["response_len"]


def attn_map_to_dist_mx(attention_map, lower_bound=0.0):
    n_tokens = attention_map.shape[1]
    distance_mx = 1 - torch.clamp(
        attention_map.to(dtype=torch.float32), min=lower_bound
    )  # torch.where(attn_mx > lower_bound, attn_mx, 0.0)

    zero_diag = torch.ones(n_tokens, n_tokens) - torch.eye(n_tokens)
    distance_mx *= zero_diag
    distance_mx = torch.minimum(distance_mx.transpose(0, 1), distance_mx)
    return distance_mx.numpy()


def exact_token_entropy(attn_map, answ_pos):
    entropy_value = 0
    for i in range(answ_pos, attn_map.shape[0]):
        row = attn_map[i]
        entropy_value += entropy(row[:answ_pos] / row[:answ_pos].sum())
    return entropy_value / (attn_map.shape[0] - answ_pos + 1)


def distance_between_clusters(distance_mx, answ_pos):
    n_tokens = distance_mx.shape[-1]
    response_len = n_tokens - answ_pos
    adj_matrix = np.zeros((response_len + 1, response_len + 1))
    adj_matrix[0, 1:] = distance_mx[answ_pos:, :answ_pos].min(-1)
    adj_matrix[1:, 0] = distance_mx[answ_pos:, :answ_pos].min(-1)
    adj_matrix[1:, 1:] = distance_mx[answ_pos:, answ_pos:]

    graph = nx.from_numpy_array(adj_matrix)
    weights = [e[-1]["weight"] for e in graph.edges(0, data=True)]

    return min(weights)


def n_edges_between_clusters(distance_mx, answ_pos):
    n_tokens = distance_mx.shape[-1]
    response_len = n_tokens - answ_pos
    adj_matrix = np.zeros((response_len + 1, response_len + 1))
    adj_matrix[0, 1:] = distance_mx[answ_pos:, :answ_pos].min(-1)
    adj_matrix[1:, 0] = distance_mx[answ_pos:, :answ_pos].min(-1)
    adj_matrix[1:, 1:] = distance_mx[answ_pos:, answ_pos:]

    graph = nx.from_numpy_array(adj_matrix)
    n_edges = len(graph.edges(0, data=True))

    return n_edges


def row_to_entropy(row, l: int, h: int):
    sample = row["entire_text"]
    input_ids = tokenizer(sample, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    answ_pos = (
        tokenizer(
            sample[: sample.rfind("A:")], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].shape[-1]
        + 1
    )
    with torch.no_grad():
        outputs = llm(input_ids.cuda(), output_attentions=True)
        attention_map = outputs.attentions[l][0][h].cpu()

    return exact_token_entropy(attention_map, answ_pos)


def row_to_cluster_distance(row, l: int, h: int):
    sample = row["entire_text"]
    input_ids = tokenizer(sample, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    answ_pos = (
        tokenizer(
            sample[: sample.rfind("A:")], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].shape[-1]
        + 1
    )
    with torch.no_grad():
        outputs = llm(input_ids.cuda(), output_attentions=True)
        attention_map = outputs.attentions[l][0][h].cpu()
        distance_mx = attn_map_to_dist_mx(attention_map=attention_map)

    return distance_between_clusters(distance_mx, answ_pos)


def plot():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plot heatmaps for the analysis of MTop-Div behaviour",
        description="Calculate and plot attention map features",
    )

    parser.add_argument("model_name")
    parser.add_argument("dataset_name")
    parser.add_argument(
        "--entropy",
        "-e",
        action="store_true",
        help="Calculate entropy of response-prompt attention",
    )
    parser.add_argument(
        "--cluster-distance",
        "-c",
        action="store_true",
        help="Calculate prompt-response cluster distance",
    )
    parser.add_argument(
        "--n-edges",
        "-n",
        action="store_true",
        help="Calculate number of edges coming from response to prompt in an MST",
    )
    parser.add_argument("--save-path", default="/app/cache/plot/")
    parser.add_argument("--mtd-path", default="/app/cache/mtopdiv")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    login(os.getenv("HUGGING_FACE_API_KEY"))

    model_name = args.model_name
    model_id, n_l = meta_dict[model_name]
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="eager"
    )
    device = args.device
    llm = llm.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds_name = args.dataset_name
    if ds_name in ["CoQA", "SQuAD"]:
        df = pd.read_csv(f"/app/data/raw/{ds_name}/{ds_name.lower()}_{model_name}.csv")
    elif ds_name == "RAGTruth_QA":
        file_path = f"/app/data/processed/RAGTruth/{model_name}/QA.json"
        df = pd.read_json(file_path)
    else:
        raise NotImplementedError
    if ds_name == "CoQA":
        df["entire_text"] = df.apply(
            lambda sample: sample["context"]
            + " Q: "
            + sample["question"]
            + " A: "
            + sample["generated_answer"],
            axis=1,
        )
        answer_token = "A:"
        if model_name == "Llama-3.1-8B-Instruct":
            df["entire_text"] = df["entire_text"].apply(
                lambda x: f"<|begin_of_text|>{x}"
            )
        elif model_name == "Llama-2-7b-chat-hf":
            df["entire_text"] = df["entire_text"].apply(lambda x: f"<s>{x}</s>")
        else:
            raise NotImplementedError
    elif ds_name == "SQuAD":
        if model_name == "Llama-3.1-8B-Instruct":

            def insert_context_question(row):
                # Assuming 'prompt' is the template where you want to insert context and question
                new_prompt = row["prompt"].format(row["context"], row["question"])
                return new_prompt

            df.rename(columns={"generated_answer": "response"}, inplace=True)
            df["prompt"] = df.apply(insert_context_question, axis=1)
            df["prompt"] = df["prompt"].apply(lambda x: f"<|begin_of_text|>{x}")
            df["response"] = df["response"].apply(lambda x: f"{x}<|eot_id|>")

            df["entire_text"] = df.apply(
                lambda sample: sample["prompt"] + sample["response"],
                axis=1,
            )
            answer_token = "Answer:"
        elif model_name == "Llama-2-7b-chat-hf":
            df.rename(columns={"generated_answer": "response"}, inplace=True)
            df["prompt"] = (
                "<s>[INST] "
                + df["prompt"]
                + "Context: "
                + df["context"]
                + "\nQuestion: "
                + df["question"]
                + "\nAnswer: [/INST]"
            )
            df["id"] = df.index
            df["response"] = df["response"].apply(lambda x: f"{x}</s>")
            df["entire_text"] = df.apply(
                lambda sample: sample["prompt"] + sample["response"],
                axis=1,
            )
            answer_token = "[/INST]"
        else:
            raise NotImplementedError
    elif ds_name == "RAGTruth_QA":
        df["entire_text"] = df.apply(
            lambda sample: sample["prompt"] + sample["response"],
            axis=1,
        )
        answer_token = "[/INST]"
        df = df[:350]
    else:
        raise NotImplementedError
    save_path = Path(args.save_path)
    # mtd_path = Path(args.mtd_path) / (ds_name.lower() + "/zero_out_prompt/" + model_name + "-text-completion")
    mtd_path = Path(args.mtd_path) / (
        ds_name.lower() + "/zero_out_prompt/" + model_name
    )
    df = process_samples_sequentially(
        df, tokenizer, llm, answer_token, n_l, mtd_path, args
    )

    df.to_csv(save_path / f"{ds_name.lower()}_{args.model_name}_full_entropy.csv")
