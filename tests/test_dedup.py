from data.pipeline.dedup import generate_hash, deduplicate


def test_generate_hash_is_deterministic():
    text = "What is Docker?"

    assert generate_hash(text) == generate_hash(text)


def test_generate_hash_distinguishes_text():
    assert generate_hash("Linux") != generate_hash("Docker")


def test_deduplicate_removes_duplicates():
    dataset = [
        {
            "id": "001",
            "input": "What is Docker?",
            "output": "Answer A",
        },
        {
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
        {"input": "What is Linux?", "output": "Answer A"},
        {"input": "What is Docker?", "output": "Answer B"},
    ]

    result = deduplicate(dataset)

    assert len(result) == 2


def test_deduplicate_handles_empty_dataset():
    assert deduplicate([]) == []
