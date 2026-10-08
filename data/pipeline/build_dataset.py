import json
from pathlib import Path


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_training_format(item):
    return {
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"]
    }


def build_training_dataset(dataset):

    training_data = []

    for item in dataset:

        training_item = convert_to_training_format(item)

        training_data.append(training_item)

    return training_data


def save_jsonl(dataset, path):

    with open(path, "w", encoding="utf-8") as f:

        for item in dataset:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )


if __name__ == "__main__":

    input_file = Path(
        "../../data/cleaned/final_dataset.json"
    )

    output_file = Path(
        "../../data/train/train.jsonl"
    )

    dataset = load_dataset(input_file)

    training_dataset = build_training_dataset(
        dataset
    )

    save_jsonl(
        training_dataset,
        output_file
    )

    print(
        f"Training dataset built. Samples: {len(training_dataset)}"
    )
