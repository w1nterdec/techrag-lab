import json

import pytest

from data.pipeline import ingest, io_utils
from data.pipeline.validation import DatasetValidationError, validate_standard_dataset


def test_raw_category_defaults_and_empty_strings():
    sample = ingest.convert_to_schema({"question": "", "answer": ""}, 3)
    assert sample["id"] == "techrag_00003"
    assert sample["metadata"]["category"] == "unknown"
    assert sample["input"] == sample["output"] == ""
    validate_standard_dataset([sample])
    explicit = ingest.convert_to_schema({"question": "q", "answer": "a", "category": ""}, 0)
    assert explicit["metadata"]["category"] == ""


@pytest.mark.parametrize("raw", [[], [{"question": "问题", "answer": "答案", "category": "linux"}]])
def test_ingest_round_trip_only_in_temporary_directory(tmp_path, raw):
    source = tmp_path / "raw.json"
    output = tmp_path / "standard.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    ingest.build_dataset(source, output)
    actual = json.loads(output.read_text(encoding="utf-8"))
    validate_standard_dataset(actual)
    assert len(actual) == len(raw)
    if raw:
        assert actual[0]["input"] == raw[0]["question"]
        assert actual[0]["output"] == raw[0]["answer"]
        assert actual[0]["metadata"]["category"] == "linux"


def test_invalid_raw_preserves_existing_temporary_output(tmp_path):
    source = tmp_path / "raw.json"
    output = tmp_path / "standard.json"
    source.write_text('[{"question":"q","answer":"a"},{"question":"q","answer":null}]')
    output.write_bytes(b"keep-existing-output")
    with pytest.raises(DatasetValidationError) as caught:
        ingest.build_dataset(source, output)
    assert (caught.value.index, caught.value.field) == (1, "answer")
    assert output.read_bytes() == b"keep-existing-output"


def test_generated_standard_data_is_checked_before_open(monkeypatch):
    monkeypatch.setattr(ingest, "load_raw_data", lambda path: [{"question": "", "answer": ""}])
    monkeypatch.setattr(ingest, "_convert_to_schema", lambda item, index: {})
    opened = []

    def forbidden_open(*args, **kwargs):
        opened.append(args)
        raise AssertionError("output opened before generated data validation")

    monkeypatch.setattr(io_utils, "_atomic_write", forbidden_open)
    with pytest.raises(DatasetValidationError):
        ingest.build_dataset("unused-input", "unused-output")
    assert opened == []
