import json
from pathlib import Path

if __package__:
    from .validation import validate_standard_dataset, validate_standard_record, validate_training_dataset
else:
    from validation import validate_standard_dataset, validate_standard_record, validate_training_dataset


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_training_format(item):
    validate_standard_record(item)
    return _convert_to_training_format(item)


def _convert_to_training_format(item):
    return {
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"]
    }


def build_training_dataset(dataset):
    validate_standard_dataset(dataset)

    training_data = []

    for item in dataset:

        training_item = _convert_to_training_format(item)

        training_data.append(training_item)

    return training_data


def save_jsonl(dataset, path):
    validate_training_dataset(dataset)

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
