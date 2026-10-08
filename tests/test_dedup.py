from pathlib import Path

from data.pipeline import dedup
from data.pipeline.quality_filter import quality_filter
from data.pipeline.dedup import generate_hash, deduplicate


def test_generate_hash_is_deterministic():
    text = "What is Docker?"

    assert generate_hash(text) == generate_hash(text)


def test_generate_hash_distinguishes_text():
    assert generate_hash("Linux") != generate_hash("Docker")


def test_deduplicate_removes_duplicates():
    dataset = [
        {
            "instruction": "回答问题",
            "metadata": {"category": "linux", "language": "zh", "source": "manual"},
            "id": "001",
            "input": "What is Docker?",
            "output": "Answer A",
        },
        {
            "instruction": "回答问题",
            "metadata": {"category": "linux", "language": "zh", "source": "manual"},
            "id": "002",
            "input": "What is Docker?",
            "output": "Answer B",
        },
    ]

    result = deduplicate(dataset)

    assert len(result) == 1
    assert result[0]["id"] == "001"


def test_deduplicate_keeps_unique_samples():
    dataset = [
        {"id": "linux", "instruction": "回答问题", "metadata": {"category": "linux", "language": "zh", "source": "manual"}, "input": "What is Linux?", "output": "Answer A"},
        {"id": "docker", "instruction": "回答问题", "metadata": {"category": "docker", "language": "zh", "source": "manual"}, "input": "What is Docker?", "output": "Answer B"},
    ]

    result = deduplicate(dataset)

    assert len(result) == 2


def test_deduplicate_handles_empty_dataset():
    assert deduplicate([]) == []


def make_entry_sample(sample_id, output):
    return {
        "id": sample_id,
        "instruction": "回答技术问题",
        "input": "如何查看 Linux 监听端口？",
        "output": output,
        "metadata": {"category": "linux", "language": "zh", "source": "manual"},
    }


def capture_main_output(monkeypatch, dataset):
    saved = []

    def fake_load(path):
        assert path == Path("../../data/cleaned/clean_dataset.json")
        return dataset

    def fake_save(records, path):
        assert path == Path("../../data/cleaned/dedup_dataset.json")
        saved.append(records)

    monkeypatch.setattr(dedup, "load_dataset", fake_load)
    monkeypatch.setattr(dedup, "save_dataset", fake_save)
    dedup.main()
    assert len(saved) == 1
    return saved[0]


def test_main_keeps_qualified_answer_after_unqualified_first(monkeypatch):
    bad = make_entry_sample("bad", "不知道")
    good = make_entry_sample("good", "可以使用 ss -tlnp 命令查看系统监听端口。")

    saved = capture_main_output(monkeypatch, [bad, good])

    assert saved == [good]
    assert quality_filter(saved) == [good]


def test_main_keeps_qualified_answer_before_unqualified_last(monkeypatch):
    good = make_entry_sample("good", "可以使用 ss -tlnp 命令查看系统监听端口。")
    bad = make_entry_sample("bad", "不知道")

    assert capture_main_output(monkeypatch, [good, bad]) == [good]


def test_main_rejects_all_unqualified_answers(monkeypatch):
    first = make_entry_sample("first", "不知道")
    second = make_entry_sample("second", "不清楚")

    assert capture_main_output(monkeypatch, [first, second]) == []


def test_main_keeps_first_of_two_qualified_answers(monkeypatch):
    first = make_entry_sample("first", "可以使用 ss -tlnp 命令查看系统监听端口。")
    second = make_entry_sample("second", "可以使用 ss -ltn 命令查看监听中的 TCP 端口。")

    assert capture_main_output(monkeypatch, [first, second]) == [first]
