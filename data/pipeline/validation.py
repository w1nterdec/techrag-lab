"""Structural contracts. Text quality remains the responsibility of filters."""

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "dataset_schema.json"
RAW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["question", "answer"],
    "properties": {key: {"type": "string"} for key in ("question", "answer", "category")},
    "additionalProperties": True,
}
TRAINING_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["instruction", "input", "output"],
    "properties": {key: {"type": "string"} for key in ("instruction", "input", "output")},
    "additionalProperties": True,
}


class DatasetValidationError(ValueError):
    """The first structural error, with a zero-based index and field path."""

    def __init__(self, index, field, message, layer):
        self.index = index
        self.field = field
        self.layer = layer
        location = layer if index is None else f"{layer}[{index}]"
        super().__init__(f"{location}.{field}: {message}")


def _checked_validator(schema):
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


_RAW_VALIDATOR = _checked_validator(RAW_SCHEMA)
_TRAINING_VALIDATOR = _checked_validator(TRAINING_SCHEMA)
_ARRAY_VALIDATOR = _checked_validator({"type": "array"})


@lru_cache(maxsize=1)
def _standard_validator():
    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        return _checked_validator(json.load(file))


def _validate(value, validator, index, layer):
    error = next(validator.iter_errors(value), None)
    if error is None:
        return
    path = list(error.absolute_path)
    if error.validator == "required":
        # Use schema order rather than parsing the library's human-readable message.
        path.append(next(key for key in error.validator_value if key not in error.instance))
    field = ".".join(str(part) for part in path) or "$"
    raise DatasetValidationError(index, field, error.message, layer) from error


def _validate_dataset(dataset, validator, layer):
    _validate(dataset, _ARRAY_VALIDATOR, None, layer)
    for index, item in enumerate(dataset):
        _validate(item, validator, index, layer)


def validate_raw_record(item, index=0):
    _validate(item, _RAW_VALIDATOR, index, "raw")


def validate_raw_dataset(dataset):
    _validate_dataset(dataset, _RAW_VALIDATOR, "raw")


def validate_standard_record(item, index=0):
    _validate(item, _standard_validator(), index, "standard")


def validate_standard_dataset(dataset):
    _validate_dataset(dataset, _standard_validator(), "standard")


def validate_training_dataset(dataset):
    _validate_dataset(dataset, _TRAINING_VALIDATOR, "training")
