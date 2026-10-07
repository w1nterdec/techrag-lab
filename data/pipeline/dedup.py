import json
import hashlib
from pathlib import Path


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )


def generate_hash(text):

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


def deduplicate(dataset):

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


if __name__ == "__main__":

    input_file = Path(
        "../../data/cleaned/clean_dataset.json"
    )

    output_file = Path(
        "../../data/cleaned/dedup_dataset.json"
    )


    dataset = load_dataset(input_file)

    dedup_dataset = deduplicate(dataset)

    save_dataset(
        dedup_dataset,
        output_file
    )

    print(
        f"Deduplication completed. Samples: {len(dedup_dataset)}"
    )
