import json
import sys

if __package__:
    from .validation import validate_standard_dataset
else:
    from validation import validate_standard_dataset


if __package__:
    from .io_utils import write_standard_json
    from .paths import parse_stage_args
else:
    from io_utils import write_standard_json
    from paths import parse_stage_args


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    if not text:
        return ""

    return text.strip()


def clean_dataset(dataset):
    validate_standard_dataset(dataset)

    cleaned = []

    for item in dataset:

        input_text = clean_text(item.get("input"))
        output_text = clean_text(item.get("output"))

        if not input_text or not output_text:
            continue

        item["input"] = input_text
        item["output"] = output_text

        cleaned.append(item)

    return cleaned


def save_dataset(dataset, path):
    write_standard_json(dataset, path)


def main(argv=()):
    input_file, output_file = parse_stage_args("clean", argv)
    dataset = load_dataset(input_file)
    result = clean_dataset(dataset)
    save_dataset(result, output_file)
    print(f"Cleaning completed. Samples: {len(result)}")


if __name__ == "__main__":
    main(sys.argv[1:])
