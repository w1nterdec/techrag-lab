import json
import hashlib
import sys

if __package__:
    from .validation import validate_standard_dataset
else:
    from validation import validate_standard_dataset

if __package__:
    from .quality_filter import quality_filter
else:
    from quality_filter import quality_filter


if __package__:
    from .io_utils import write_standard_json
    from .paths import parse_stage_args
else:
    from io_utils import write_standard_json
    from paths import parse_stage_args


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset, path):
    write_standard_json(dataset, path)


def generate_hash(text):

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


def deduplicate(dataset):
    validate_standard_dataset(dataset)

    seen = set()
    results = []

    for item in dataset:

        text_hash = generate_hash(
            item["input"]
        )

        if text_hash in seen:
            continue

        seen.add(text_hash)
        results.append(item)

    return results


def main(argv=()):
    input_file, output_file = parse_stage_args("dedup", argv)
    dataset = load_dataset(input_file)
    qualified_dataset = quality_filter(dataset)
    dedup_dataset = deduplicate(qualified_dataset)
    save_dataset(dedup_dataset, output_file)
    print(f"Deduplication completed. Samples: {len(dedup_dataset)}")


if __name__ == "__main__":
    main(sys.argv[1:])
