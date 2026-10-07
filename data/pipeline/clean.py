import json
from pathlib import Path


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    if not text:
        return ""

    return text.strip()


def clean_dataset(dataset):

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

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    input_file = Path("../../data/cleaned/dataset.json")
    output_file = Path("../../data/cleaned/clean_dataset.json")

    dataset = load_dataset(input_file)

    cleaned_dataset = clean_dataset(dataset)

    save_dataset(
        cleaned_dataset,
        output_file
    )

    print(
        f"Cleaning completed. Samples: {len(cleaned_dataset)}"
    )
