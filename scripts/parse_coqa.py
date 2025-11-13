import argparse
import json
from collections import defaultdict

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="assets/data/coqa")
    args = parser.parse_args()

    with open(f"{args.data_dir}/coqa-dev-v1.0.json", "r") as f:
        data = json.load(f)["data"]

    dataset = defaultdict(list)

    for sample_id, sample in enumerate(data):
        story = sample["story"]
        questions = sample["questions"]
        answers = sample["answers"]
        additional_answers = sample["additional_answers"]

        for question_index, question in enumerate(questions):
            dataset["context"].append(story)
            dataset["question"].append(question["input_text"])
            additional_answers_list = []
            for i in range(3):
                additional_answers_list.append(
                    additional_answers[str(i)][question_index]["input_text"]
                )

            dataset["answers"].append(
                [answers[question_index]["input_text"]] + additional_answers_list
            )
            dataset["id"].append(sample["id"] + "_" + str(question_index))
            dataset["context_id"].append(sample["id"])
            story = (
                story
                + " Q: "
                + question["input_text"]
                + " A: "
                + answers[question_index]["input_text"]
            )
            if not story[-1] == ".":
                story = story + "."

    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset_df.to_json(
        f"{args.data_dir}/preprocessed.json", orient="records", lines=True
    )
