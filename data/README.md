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

The planned workflow:

Raw Data

    |
    v

Ingestion

    |
    v

Cleaning

    |
    v

Deduplication

    |
    v

Quality Filtering

    |
    v

Training Dataset


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
