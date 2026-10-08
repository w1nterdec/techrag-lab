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
    "language": "zh",
    "source": "manual"
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

Structural validation runs before atomic output creation. Default data paths are
now resolved from the pipeline modules' location rather than the working directory.

---


## Execution and Atomic Outputs

The five standalone stages remain available as modules or direct scripts. They
share `--input` and `--output` options. Defaults refer to the same files under
this repository's `data/` directory, regardless of the current working directory.
Explicit relative paths are resolved against the caller's working directory.
Calling a stage's `main()` without arguments uses defaults; command-line entry
points pass their arguments explicitly.

For isolated execution from the repository root:

```bash
python -m data.pipeline.ingest --input /tmp/raw.json --output /tmp/techrag/dataset.json
python -m data.pipeline.clean --input /tmp/techrag/dataset.json --output /tmp/techrag/clean.json
python -m data.pipeline.dedup --input /tmp/techrag/clean.json --output /tmp/techrag/dedup.json
python -m data.pipeline.quality_filter --input /tmp/techrag/dedup.json --output /tmp/techrag/final.json
python -m data.pipeline.build_dataset --input /tmp/techrag/final.json --output /tmp/techrag/train.jsonl
```

The unified entry point loads Raw JSON once, passes data in memory through
Ingest → Clean → Quality Pre-filter → Dedup → Quality Recheck → Training Build,
and atomically publishes only `<output-dir>/train.jsonl`. It does not generate
or read intermediate JSON artifacts. `ingest_records()` provides the in-memory
Raw-to-Standard conversion; `build_dataset(raw_path, output_path)` remains available.

```bash
python -m data.pipeline.run_pipeline --input /tmp/raw.json --output-dir /tmp/techrag-output
```

`--input` is optional and defaults to this repository's `data/raw/sample.json`.
`--output-dir` is required: omitting it exits with an argument error and does not
fall back to the formal training directory. Existing output files are replaced
on a successful run. There are no dry-run or report-file options.

From another working directory, direct stage scripts can be invoked by absolute
path. For `python -m`, the repository must be on Python's import path, for example:

```bash
PYTHONPATH=/path/to/techrag-lab /path/to/venv/bin/python -m data.pipeline.run_pipeline --input /tmp/raw.json --output-dir /tmp/techrag-output
```

The runner emits a JSON report to stdout on success and stderr on pipeline failure.
Its exit status is 0 on success, 1 on pipeline failure, and 2 for argument errors.
The report includes six stages' input/output sample counts, elapsed seconds and
status (`completed`, `failed`, `not_run`), plus the failed operation and original
exception type/message. Counts are null when unavailable. A failed Training save
can include `built_count` while `output_count` remains null. Training timing includes
building and saving. `output_committed` becomes true only after final replacement.
Programmatic `run_pipeline(input_path, output_dir)` returns the report or raises
`PipelineExecutionError` with `.stage`, `.report` and the original exception as
`__cause__`. Dataset validation failures retain `index`, `field`, `layer` in the report.

`io_utils.py` performs complete structure validation before creating directories
or temporary files. JSON and JSONL keep their existing serialization formats.
A uniquely named temporary file is created in the target directory, written,
flushed, fsynced and closed before `os.replace()` publishes it. Serialization and
I/O failures leave the old target unchanged and attempt temporary-file cleanup.
Cleanup failures are attached as notes to the original exception; they do not
replace it. Such failures can leave a temporary file for manual cleanup.

This guarantees atomic replacement of one file, not a transaction across the
standalone stages' multiple outputs. It does not guarantee full power-loss
durability, concurrent-run coordination, or preservation of target permissions,
ACLs or symbolic-link behavior. Forced termination may leave temporary files.

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
