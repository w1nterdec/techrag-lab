import json
import sys
from datetime import datetime

if __package__:
    from .validation import validate_raw_dataset, validate_raw_record, validate_standard_dataset
else:
    from validation import validate_raw_dataset, validate_raw_record, validate_standard_dataset


if __package__:
    from .io_utils import write_standard_json
    from .paths import parse_stage_args
else:
    from io_utils import write_standard_json
    from paths import parse_stage_args


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


def ingest_records(raw_data):
    validate_raw_dataset(raw_data)
    dataset = [_convert_to_schema(item, index) for index, item in enumerate(raw_data)]
    validate_standard_dataset(dataset)
    return dataset


def build_dataset(raw_path, output_path):
    dataset = ingest_records(load_raw_data(raw_path))
    write_standard_json(dataset, output_path)


def main(argv=()):
    input_file, output_file = parse_stage_args("ingest", argv)
    build_dataset(input_file, output_file)
    print("Dataset ingestion completed.")


if __name__ == "__main__":
    main(sys.argv[1:])
