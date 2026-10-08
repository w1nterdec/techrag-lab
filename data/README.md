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
