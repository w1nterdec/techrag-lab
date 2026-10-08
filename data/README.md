# Dataset

Dataset engineering workspace for TechRAG-Lab.

This directory contains raw data, processed datasets, schemas and data processing pipelines.

---

## Directory Structure

data/

├── raw/
│   └── Original collected data
│       - Documentation
│       - GitHub issues
│       - Personal experiment records

├── cleaned/
│   └── Processed and filtered datasets
│       Used for training and evaluation

├── pipeline/
│   └── Data processing scripts
│       - ingestion
│       - cleaning
│       - deduplication
│       - quality filtering

└── schema/
    └── Dataset format definition


---

## Dataset Format

All samples follow the unified dataset schema.

Required fields:

- id
- instruction
- input
- output
- metadata


Example structure:

{
  "id": "docker_00001",
  "instruction": "Analyze the error and provide a solution",
  "input": "WSL2 docker ps returns permission denied",
  "output": "The user does not have permission to access Docker daemon socket.",
  "metadata": {
    "category": "docker",
    "language": "zh"
  }
}


---

## Data Pipeline

The current workflow:

Raw Data

    |
    v

Ingestion

    |
    v

Cleaning

    |
    v

Quality Filtering (inside dedup.py)

    |
    v

Deduplication

    |
    v

Quality Recheck (quality_filter.py)

    |
    v

Training Dataset

`dedup.py` reads `cleaned/clean_dataset.json`, applies the existing
`quality_filter()` before `deduplicate()`, and writes `cleaned/dedup_dataset.json`.
Deduplication still uses the input text and keeps the first qualified sample for
identical inputs. `quality_filter.py` rechecks this result and writes
`cleaned/final_dataset.json`; `build_dataset.py` exports it to `train/train.jsonl`.


---


## Structural Contracts and Validation

`pipeline/validation.py` executes Raw, Standard and Training contracts with
`jsonschema==4.26.0`. `schema/dataset_schema.json` is a Draft 2020-12 JSON Schema
for one Standard sample, loaded relative to the validation module. Every Schema
is checked with `Draft202012Validator.check_schema()` before use; the Standard
validator is cached after its first load.

All dataset inputs must be arrays of objects. Empty arrays are valid.

- Raw records require string `question` and `answer`. Optional `category` must
  be a string when present; ingestion defaults a missing category to `unknown`.
- Standard records require string `id`, `instruction`, `input`, `output`, and
  object `metadata`. Metadata requires string `category`, `language`, `source`.
- Training records require string `instruction`, `input`, `output`.
  `build_training_dataset()` exports exactly these three fields.

Empty and whitespace-only strings are structurally valid. Nulls, numbers,
booleans, arrays and objects cannot replace strings. Additional fields are
allowed by validation. Existing transformations retain their behavior, including
training projection; validation itself does not change records.

Public dataset processing functions validate the complete input before processing.
Public single-record converters validate their input too. Ingestion also validates
the generated Standard dataset. Every save function validates all records before
opening its output file, including records later in the array.

Structural errors raise `DatasetValidationError`, a `ValueError` with `index`,
`field` and `layer` attributes, for example `standard[1].metadata.source`.
Indices are zero-based. Top-level errors use `index=None` and field `$`.
The first error stops processing; invalid records are not silently filtered.

`load_*()` functions only parse JSON. Low-level `clean_text()` and
`check_quality()` retain their existing helper behavior, including
`clean_text(None) == ""`; they do not validate complete Standard records.
Dataset entry points enforce the contract before calling these helpers.
Cleaning still removes empty input/output, and quality filtering retains the
existing minimum lengths of 5 input characters and 10 output characters after
stripping whitespace. Pre-filtering before deduplication and the final quality
recheck remain in place.

Validation does not provide atomic writes or remove the scripts' existing
working-directory-dependent input/output paths.

---


## Data Sources

Initial dataset sources:

- Official documentation
- GitHub issues
- Technical articles
- Personal engineering experiments


---

## Version

Current dataset specification:

v0.1
