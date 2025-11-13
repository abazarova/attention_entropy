import argparse
import re
from pathlib import Path

import evaluate
import pandas as pd
from num2words import num2words

rouge = evaluate.load("rouge")


def str_to_list(answers: str):
    return [elem.strip("' ") for elem in answers.rstrip("]").lstrip("[").split(", ")]


def calc_max_rougel(gen_answer, answers):
    gen_answer = gen_answer.rstrip("Q:")
    answer_list = str_to_list(answers)
    rougel = rouge.compute(
        predictions=[gen_answer] * len(answer_list),
        references=answer_list,
        use_aggregator=False,
    )["rougeL"]
    return max(rougel)


def generate_alias(answer: str):
    """Create an alias for the answer with the numbers written as words."""
    return re.sub(r"[0-9]+", repl=lambda x: num2words(x.group()), string=answer)


def is_substring(gen_answer, answers):
    gen_answer = gen_answer.rstrip("Q:")
    answer_alias = generate_alias(gen_answer)
    answer_list = str_to_list(answers)
    return (
        any(answ in gen_answer for answ in answer_list)
        or any(answ in answer_alias for answ in answer_list)
        or any(answer_alias in answ for answ in answer_list)
        or any(gen_answer in answ for answ in answer_list)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument(
        "dataset",
        type=str,
        choices=["SQuAD", "CoQA", "XSum", "HotpotQA", "NQ_Swap"],
    )
    parser.add_argument("--data_dir", type=str, default="/app/data/fixed/")

    args = parser.parse_args()
    root_dir = Path(args.data_dir)
    model_name = args.model_id.split("/")[1]
    df = pd.read_csv(root_dir / args.dataset / model_name / "annotation.csv")
    df = df.dropna()
    n_hallucinated, n_grounded = (
        len(df[df["hallucination"].str.contains("Yes", na=False)]),
        len(df[df["hallucination"].str.contains("No", na=False)]),
    )

    print(
        f"Annotation statistics:\nTOTAL: {len(df)}\n"
        + f"Hallucinated: {n_hallucinated}\n"
        + f"Grounded: {n_grounded}\n"
        + f"Hallucinated + grounded: {n_hallucinated + n_grounded}"
    )

    df["hallucination"] = df["hallucination"].apply(lambda x: 1 if "Yes" in x else 0)
    if args.dataset in ["SQuAD", "CoQA"]:
        df["rougel"] = df.apply(
            lambda x: calc_max_rougel(x["generated_answer"], x["answers"]), axis=1
        )
        df["is_substring"] = df.apply(
            lambda x: is_substring(x["generated_answer"], x["answers"]), axis=1
        )

        df["final_hallucination"] = df.apply(
            lambda x: (x["hallucination"] == 1)
            and (not x["is_substring"])
            and (x["rougel"] < 0.3),
            axis=1,
        ).astype(int)

        size = len(df[df["final_hallucination"] == 1])

        final_dataset = pd.concat(
            (
                df[(df["rougel"] > 0.95) & (df["hallucination"] == 0)].sample(size),
                df[df["final_hallucination"] == 1],
            )
        )
        final_dataset["hallucination"] = final_dataset["final_hallucination"]

        if args.dataset == "SQuAD":
            final_dataset["prompt"] = (
                "Given the context, answer the question in a single brief but complete sentence."
                + "Note that your answer should be strictly based on the given context."
                + "In case the context does not contain the necessary information to answer the question, "
                + 'please reply with: "Unable to answer based on given context. "'
                + "Context: {}\n"
                + "Question: {}\n"
                + "Answer: "
            )
        final_dataset = final_dataset.drop(
            columns=["final_hallucination", "Unnamed: 0", "Unnamed: 0.1"]
        )
    elif args.dataset == "XSum":
        size = len(df[df["hallucination"] == 1])
        final_dataset = pd.concat(
            (
                df[df["hallucination"] == 0].sample(size),
                df[df["hallucination"] == 1],
            )
        )
        final_dataset["prompt"] = (
            "Article: {}\nSummarize the article in one sentence. Summary:"
        )
        final_dataset = final_dataset.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

    else:
        raise NotImplementedError

    print(
        f"Final statistics:\nTOTAL: {len(final_dataset)}\n"
        + f"Hallucinated: {len(final_dataset[final_dataset['hallucination'] == 1])}\n"
        + f"Grounded: {len(final_dataset[final_dataset['hallucination'] == 0])}\n"
    )
    final_dataset = final_dataset.sample(len(final_dataset))
    final_dataset.to_csv(
        root_dir / args.dataset / f"{args.dataset.lower()}_{model_name}.csv"
    )
