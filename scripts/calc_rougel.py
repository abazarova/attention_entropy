import argparse
import ast
from pathlib import Path

import pandas as pd


def rouge_precision_substring(candidate, reference) -> float:
    """
    ROUGE precision implementation that returns 1 if candidate is a substring of reference,
    otherwise returns the proportion of candidate that matches.

    Args:
        candidate (str): The generated text
        reference (str): The reference text

    Returns:
        float: 1.0 if candidate is substring of reference, otherwise proportion of match
    """
    candidate_lower = candidate.lower().strip()
    reference_lower = reference.lower().strip()

    # If candidate is empty, precision is undefined (return 0)
    if not candidate_lower:
        return 0.0

    # Case 1: Candidate is exact substring of reference -> precision = 1
    if candidate_lower in reference_lower:
        return 1.0

    # Case 2: Check for partial substring matches
    # Find the longest contiguous substring of candidate that appears in reference
    max_match_length = 0
    candidate_len = len(candidate_lower)

    # Check all possible substrings of candidate
    for i in range(candidate_len):
        for j in range(i + 1, candidate_len + 1):
            substring = candidate_lower[i:j]
            if substring in reference_lower:
                max_match_length = max(max_match_length, j - i)

    # Precision = (length of longest matching substring) / (length of candidate)
    return max_match_length / candidate_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument(
        "dataset",
        type=str,
        choices=["SQuAD", "CoQA"],
    )
    parser.add_argument("--data_dir", type=str, default="/app/data/fixed/")

    args = parser.parse_args()

    model_id = args.model_id
    model_name = model_id.split("/")[1]
    dataset = args.dataset
    data_dir = args.data_dir

    root_dir = Path(data_dir) / dataset / model_name
    load_path = root_dir / "generations.csv"

    df = pd.read_csv(load_path)
    print(f"Original dataset size: {len(df)}")
    df = df[df["generated_answer"].notna()]
    print(f"After nan filtering: {len(df)}")
    rougel_scores = []
    for idx, row in df.iterrows():
        answers = ast.literal_eval(row["answers"])
        generated_answer = row["generated_answer"]

        rougel = max(
            [rouge_precision_substring(answ, generated_answer) for answ in answers]
        )
        rougel_scores.append(rougel)

    df["rougel"] = rougel_scores

    df.to_csv(root_dir / "generations_with_rougel.csv", index=False)
