import json
import sys

if __package__:
    from .validation import validate_standard_dataset, validate_standard_record, validate_training_dataset
else:
    from validation import validate_standard_dataset, validate_standard_record, validate_training_dataset


if __package__:
    from .io_utils import write_training_jsonl
    from .paths import parse_stage_args
else:
    from io_utils import write_training_jsonl
    from paths import parse_stage_args


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
    write_training_jsonl(dataset, path)


def main(argv=()):
    input_file, output_file = parse_stage_args("build_dataset", argv)
    dataset = load_dataset(input_file)
    result = build_training_dataset(dataset)
    save_jsonl(result, output_file)
    print(f"Training dataset built. Samples: {len(result)}")


if __name__ == "__main__":
    main(sys.argv[1:])
