import copy
import json

import pytest
from jsonschema import Draft202012Validator, SchemaError

from data.pipeline import validation as v


CONTRACTS = [v.validate_raw_dataset, v.validate_standard_dataset, v.validate_training_dataset]
BAD_STRINGS = [None, 1, True, [], {}]


@pytest.mark.parametrize("validate", CONTRACTS)
def test_empty_dataset_is_valid(validate):
    assert validate([]) is None


@pytest.mark.parametrize("validate", CONTRACTS)
@pytest.mark.parametrize("value", [None, 1, True, "", {}, ()])
def test_top_level_must_be_array(validate, value):
    with pytest.raises(v.DatasetValidationError) as caught:
        validate(value)
    assert caught.value.index is None
    assert caught.value.field == "$"


@pytest.mark.parametrize("validate", CONTRACTS)
@pytest.mark.parametrize("value", [None, 1, True, "", []])
def test_records_must_be_objects(validate, value):
    with pytest.raises(v.DatasetValidationError) as caught:
        validate([value])
    assert caught.value.index == 0
    assert caught.value.field == "$"


@pytest.mark.parametrize("field", ["question", "answer", "category"])
@pytest.mark.parametrize("value", BAD_STRINGS)
def test_raw_field_types(field, value):
    item = {"question": "", "answer": ""}
    item[field] = value
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_raw_dataset([{"question": "", "answer": ""}, item])
    assert (caught.value.index, caught.value.field) == (1, field)


@pytest.mark.parametrize("field", ["question", "answer"])
def test_raw_required_fields(field):
    item = {"question": "", "answer": ""}
    del item[field]
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_raw_record(item, 4)
    assert (caught.value.index, caught.value.field) == (4, field)


@pytest.mark.parametrize("field", ["id", "instruction", "input", "output"])
@pytest.mark.parametrize("value", BAD_STRINGS)
def test_standard_string_types(standard_sample, field, value):
    standard_sample[field] = value
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_record(standard_sample)
    assert (caught.value.index, caught.value.field) == (0, field)


@pytest.mark.parametrize("field", ["id", "instruction", "input", "output", "metadata"])
def test_standard_required_fields(standard_sample, field):
    del standard_sample[field]
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_dataset([standard_sample])
    assert caught.value.field == field


@pytest.mark.parametrize("value", [None, 1, True, [], "metadata"])
def test_metadata_must_be_object(standard_sample, value):
    standard_sample["metadata"] = value
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_record(standard_sample)
    assert caught.value.field == "metadata"


@pytest.mark.parametrize("field", ["category", "language", "source"])
@pytest.mark.parametrize("value", BAD_STRINGS)
def test_metadata_string_types(standard_sample, field, value):
    standard_sample["metadata"][field] = value
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_record(standard_sample)
    assert caught.value.field == "metadata." + field


@pytest.mark.parametrize("field", ["category", "language", "source"])
def test_metadata_required_fields(standard_sample, field):
    del standard_sample["metadata"][field]
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_record(standard_sample)
    assert caught.value.field == "metadata." + field


def test_empty_metadata_is_invalid(standard_sample):
    standard_sample["metadata"] = {}
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_record(standard_sample)
    assert caught.value.field == "metadata.category"


@pytest.mark.parametrize("field", ["instruction", "input", "output"])
@pytest.mark.parametrize("value", BAD_STRINGS)
def test_training_string_types(field, value):
    item = {"instruction": "", "input": "", "output": ""}
    item[field] = value
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_training_dataset([item])
    assert caught.value.field == field


@pytest.mark.parametrize("field", ["instruction", "input", "output"])
def test_training_required_fields(field):
    item = {"instruction": "", "input": "", "output": ""}
    del item[field]
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_training_dataset([item])
    assert caught.value.field == field


def test_empty_strings_and_extra_fields_are_valid_and_unchanged():
    raw = {"question": "", "answer": "  ", "category": "", "extra": [1]}
    sample = {"id": "", "instruction": "", "input": "", "output": "  ",
              "metadata": {"category": "", "language": "", "source": "", "url": ""},
              "extra": {"a": 1}}
    training = {"instruction": "", "input": "", "output": "", "extra": []}
    before = copy.deepcopy([raw, sample, training])
    v.validate_raw_dataset([raw])
    v.validate_standard_dataset([sample])
    v.validate_training_dataset([training])
    assert [raw, sample, training] == before


def test_first_bad_record_is_reported(standard_sample):
    bad = copy.deepcopy(standard_sample)
    bad["output"] = None
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_dataset([standard_sample, bad, None])
    assert (caught.value.index, caught.value.field) == (1, "output")
    assert "standard[1].output" in str(caught.value)


def test_schema_documents_are_valid():
    schema = json.loads(v.SCHEMA_PATH.read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    for document in [schema, v.RAW_SCHEMA, v.TRAINING_SCHEMA]:
        Draft202012Validator.check_schema(document)


@pytest.fixture
def temporary_schema(monkeypatch, tmp_path):
    path = tmp_path / "schema.json"
    monkeypatch.setattr(v, "SCHEMA_PATH", path)
    v._standard_validator.cache_clear()
    yield path
    v._standard_validator.cache_clear()


def test_schema_file_really_controls_validation(temporary_schema, standard_sample):
    schema = {"type": "object", "required": ["marker"],
              "properties": {"marker": {"type": "string"}}}
    temporary_schema.write_text(json.dumps(schema))
    with pytest.raises(v.DatasetValidationError) as caught:
        v.validate_standard_record(standard_sample)
    assert caught.value.field == "marker"


def test_invalid_schema_is_checked_before_use(temporary_schema):
    temporary_schema.write_text('{"type": "not-a-json-type"}')
    with pytest.raises(SchemaError):
        v.validate_standard_dataset([])
