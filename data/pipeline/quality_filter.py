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



def save_dataset(dataset, path):
    write_standard_json(dataset, path)


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



def main(argv=()):
    input_file, output_file = parse_stage_args("quality_filter", argv)
    dataset = load_dataset(input_file)
    result = quality_filter(dataset)
    save_dataset(result, output_file)
    print(f"Quality filtering completed. Samples: {len(result)}")


if __name__ == "__main__":
    main(sys.argv[1:])
