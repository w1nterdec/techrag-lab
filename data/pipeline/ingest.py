import json
from pathlib import Path
from datetime import datetime

if __package__:
    from .validation import validate_raw_dataset, validate_raw_record, validate_standard_dataset
else:
    from validation import validate_raw_dataset, validate_raw_record, validate_standard_dataset


def load_raw_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_schema(item, index):
    validate_raw_record(item, index)
    return _convert_to_schema(item, index)


def _convert_to_schema(item, index):
    return {
        "id": f"techrag_{index:05d}",
        "instruction": "Answer the technical question based on the provided information.",
        "input": item["question"],
        "output": item["answer"],
        "metadata": {
            "category": item.get("category", "unknown"),
            "language": "zh",
            "source": "manual",
            "created_at": datetime.now().isoformat()
        }
    }


def build_dataset(raw_path, output_path):

    raw_data = load_raw_data(raw_path)
    validate_raw_dataset(raw_data)

    dataset = []

    for idx, item in enumerate(raw_data):
        dataset.append(
            _convert_to_schema(item, idx)
        )

    validate_standard_dataset(dataset)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    raw_file = Path("../../data/raw/sample.json")
    output_file = Path("../../data/cleaned/dataset.json")

    build_dataset(
        raw_file,
        output_file
    )

    print("Dataset ingestion completed.")
