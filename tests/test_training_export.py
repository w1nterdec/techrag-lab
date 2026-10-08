import json

import pytest

from data.pipeline import build_dataset, clean, dedup, quality_filter
from data.pipeline.validation import DatasetValidationError


@pytest.mark.parametrize("module", [clean, dedup, quality_filter])
def test_standard_save_round_trip(tmp_path, standard_sample, module):
    standard_sample["extra"] = {"retained": True}
    path = tmp_path / "standard.json"
    module.save_dataset([standard_sample], path)
    assert json.loads(path.read_text(encoding="utf-8")) == [standard_sample]


@pytest.mark.parametrize("save", [clean.save_dataset, dedup.save_dataset,
                                  quality_filter.save_dataset, build_dataset.save_jsonl])
def test_invalid_later_record_preserves_existing_output(tmp_path, standard_sample, save):
    path = tmp_path / "output"
    path.write_bytes(b"keep-existing-output")
    bad = dict(standard_sample, output=None)
    with pytest.raises(DatasetValidationError) as caught:
        save([standard_sample, bad], path)
    assert (caught.value.index, caught.value.field) == (1, "output")
    assert path.read_bytes() == b"keep-existing-output"


def test_training_export_is_three_fields_with_unicode_and_newlines(tmp_path, standard_sample):
    standard_sample["output"] = "中文答案\n第二行"
    standard_sample["extra"] = "not exported"
    expected = {key: standard_sample[key] for key in ("instruction", "input", "output")}
    assert build_dataset.convert_to_training_format(standard_sample) == expected
    rows = build_dataset.build_training_dataset([standard_sample])
    assert rows == [expected]
    path = tmp_path / "train.jsonl"
    build_dataset.save_jsonl(rows, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == expected
    assert set(json.loads(lines[0])) == {"instruction", "input", "output"}


def test_empty_training_export(tmp_path):
    assert build_dataset.build_training_dataset([]) == []
    path = tmp_path / "train.jsonl"
    build_dataset.save_jsonl([], path)
    assert path.read_bytes() == b""


@pytest.mark.parametrize("process", [clean.clean_dataset, quality_filter.quality_filter,
                                     dedup.deduplicate, build_dataset.build_training_dataset])
def test_processing_rejects_non_array_top_level(process):
    with pytest.raises(DatasetValidationError) as caught:
        process({})
    assert caught.value.index is None
    assert caught.value.field == "$"


@pytest.mark.parametrize("process", [clean.clean_dataset, quality_filter.quality_filter,
                                     dedup.deduplicate, build_dataset.build_training_dataset])
def test_processing_rejects_invalid_metadata_instead_of_skipping(process, standard_sample):
    standard_sample["metadata"] = {}
    with pytest.raises(DatasetValidationError) as caught:
        process([standard_sample])
    assert caught.value.field == "metadata.category"


def test_empty_text_is_structural_not_quality_error(standard_sample):
    standard_sample["input"] = ""
    assert clean.clean_dataset([dict(standard_sample)]) == []
    assert quality_filter.quality_filter([standard_sample]) == []
    assert build_dataset.build_training_dataset([standard_sample])[0]["input"] == ""
