
import hashlib
import json

import pytest

from evaluation.run_baseline import (
    MODEL_ID,
    MODEL_REVISION,
    load_dataset,
)


def test_dataset_load_and_hash(tmp_path):
    samples = [
        {
            "id": "linux_001",
            "category": "linux",
            "question": "如何查看监听端口？",
        },
        {
            "id": "docker_001",
            "category": "docker",
            "question": "如何检查 Docker 服务？",
        },
    ]

    path = tmp_path / "questions.json"
    content = json.dumps(
        samples,
        ensure_ascii=False,
    ).encode("utf-8")

    path.write_bytes(content)

    actual, digest = load_dataset(path)

    assert actual == samples
    assert digest == hashlib.sha256(content).hexdigest()


def test_duplicate_id_rejected(tmp_path):
    samples = [
        {"id": "same", "question": "Question A"},
        {"id": "same", "question": "Question B"},
    ]

    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(samples),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate sample id"):
        load_dataset(path)


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [{"id": "", "question": "Test"}],
        [{"id": "valid", "question": ""}],
        [{"id": "valid"}],
        ["not an object"],
    ],
)
def test_invalid_dataset_rejected(tmp_path, samples):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(samples),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_dataset(path)


def test_model_revision_is_pinned():
    assert MODEL_ID == "Qwen/Qwen3-0.6B"
    assert len(MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in MODEL_REVISION)


def test_baseline_dataset_contract():
    from evaluation.run_baseline import DEFAULT_DATASET

    samples, digest = load_dataset(DEFAULT_DATASET)

    assert len(samples) == 5
    assert len(digest) == 64

    ids = [item["id"] for item in samples]
    assert len(ids) == len(set(ids))
