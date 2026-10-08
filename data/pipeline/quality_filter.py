import json
from pathlib import Path

if __package__:
    from .validation import validate_standard_dataset
else:
    from validation import validate_standard_dataset


def load_dataset(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_dataset(dataset, path):
    validate_standard_dataset(dataset)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )



def check_quality(item):

    input_text = item.get("input", "")
    output_text = item.get("output", "")
    metadata = item.get("metadata")


    # 输入不能为空
    if len(input_text.strip()) < 5:
        return False


    # 输出不能为空且长度合理
    if len(output_text.strip()) < 10:
        return False


    # metadata必须存在
    if not metadata:
        return False


    return True



def quality_filter(dataset):
    validate_standard_dataset(dataset)

    filtered = []

    for item in dataset:

        if check_quality(item):
            filtered.append(item)


    return filtered



if __name__ == "__main__":


    input_file = Path(
        "../../data/cleaned/dedup_dataset.json"
    )


    output_file = Path(
        "../../data/cleaned/final_dataset.json"
    )


    dataset = load_dataset(input_file)


    filtered_dataset = quality_filter(dataset)


    save_dataset(
        filtered_dataset,
        output_file
    )


    print(
        f"Quality filtering completed. Samples: {len(filtered_dataset)}"
    )
