from data.pipeline.quality_filter import (
    check_quality,
    quality_filter,
)


def make_sample(
    input_text="如何查看 Linux 监听端口？",
    output_text="可以使用 ss -tlnp 命令查看系统监听端口。",
    metadata=None,
):
    if metadata is None:
        metadata = {"category": "linux"}

    return {
        "input": input_text,
        "output": output_text,
        "metadata": metadata,
    }


def test_valid_sample_passes():
    sample = make_sample()

    assert check_quality(sample) is True


def test_short_input_is_rejected():
    sample = make_sample(input_text="Hi")

    assert check_quality(sample) is False


def test_short_output_is_rejected():
    sample = make_sample(output_text="ok")

    assert check_quality(sample) is False


def test_missing_metadata_is_rejected():
    sample = make_sample()
    del sample["metadata"]

    assert check_quality(sample) is False


def test_empty_metadata_is_rejected():
    sample = make_sample(metadata={})

    assert check_quality(sample) is False


def test_quality_filter_keeps_only_valid_samples():
    dataset = [
        make_sample(),
        make_sample(input_text="Hi"),
        make_sample(output_text="ok"),
        make_sample(metadata={}),
    ]

    result = quality_filter(dataset)

    assert len(result) == 1
    assert result[0]["input"] == "如何查看 Linux 监听端口？"


def test_empty_dataset():
    assert quality_filter([]) == []
