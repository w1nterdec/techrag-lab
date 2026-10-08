import copy

import pytest

from data.pipeline import ingest, clean, quality_filter, dedup, build_dataset


def standard():
    return {
        "id": "sample", "instruction": "回答问题", "input": "如何查看 Linux 端口？",
        "output": "可以使用 ss -tlnp 查看系统监听端口。",
        "metadata": {"category": "linux", "language": "zh", "source": "manual"},
    }


@pytest.mark.parametrize("process", [clean.clean_dataset, quality_filter.quality_filter,
                                     dedup.deduplicate, build_dataset.build_training_dataset])
@pytest.mark.parametrize("missing", [False, True])
def test_processing_rejects_bad_later_record_before_mutation(process, missing):
    first = standard()
    first["input"] = "  " + first["input"] + "  "
    before = copy.deepcopy(first)
    bad = standard()
    if missing:
        del bad["output"]
    else:
        bad["output"] = None
    with pytest.raises(ValueError, match=r"standard\[1\]\.output"):
        process([first, bad])
    assert first == before


@pytest.mark.parametrize("module,save", [
    (clean, clean.save_dataset), (quality_filter, quality_filter.save_dataset),
    (dedup, dedup.save_dataset), (build_dataset, build_dataset.save_jsonl),
])
def test_save_validates_later_record_before_open(monkeypatch, module, save):
    bad = standard()
    bad["output"] = False
    opened = []

    def forbidden_open(*args, **kwargs):
        opened.append(args)
        raise AssertionError("output opened before validation")

    monkeypatch.setattr(module, "open", forbidden_open, raising=False)
    with pytest.raises(ValueError, match=r"(?:standard|training)\[1\]\.output"):
        save([standard(), bad], "unused-output")
    assert opened == []


@pytest.mark.parametrize("bad", [{"question": "问题", "answer": 3}, {"question": "问题"}])
def test_ingest_validates_before_output_open(monkeypatch, bad):
    monkeypatch.setattr(ingest, "load_raw_data", lambda path: [
        {"question": "问题", "answer": "答案"}, bad,
    ])
    opened = []

    def forbidden_open(*args, **kwargs):
        opened.append(args)
        raise AssertionError("output opened before validation")

    monkeypatch.setattr(ingest, "open", forbidden_open, raising=False)
    with pytest.raises(ValueError, match=r"raw\[1\]\.answer"):
        ingest.build_dataset("unused-input", "unused-output")
    assert opened == []


def test_public_raw_conversion_rejects_invalid_type():
    with pytest.raises(ValueError, match=r"raw\[7\]\.question"):
        ingest.convert_to_schema({"question": [], "answer": "答案"}, 7)


def test_public_training_conversion_rejects_missing_field():
    item = standard()
    del item["instruction"]
    with pytest.raises(ValueError, match=r"standard\[0\]\.instruction"):
        build_dataset.convert_to_training_format(item)
