"""Module-relative defaults and shared arguments for standalone stages."""

import argparse
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = {
    "ingest": (DATA_DIR / "raw/sample.json", DATA_DIR / "cleaned/dataset.json"),
    "clean": (DATA_DIR / "cleaned/dataset.json", DATA_DIR / "cleaned/clean_dataset.json"),
    "dedup": (DATA_DIR / "cleaned/clean_dataset.json", DATA_DIR / "cleaned/dedup_dataset.json"),
    "quality_filter": (DATA_DIR / "cleaned/dedup_dataset.json", DATA_DIR / "cleaned/final_dataset.json"),
    "build_dataset": (DATA_DIR / "cleaned/final_dataset.json", DATA_DIR / "train/train.jsonl"),
}


def parse_stage_args(stage, argv):
    default_input, default_output = DEFAULT_PATHS[stage]
    parser = argparse.ArgumentParser(description=f"TechRAG-Lab {stage} stage")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)
    return args.input.expanduser().resolve(), args.output.expanduser().resolve()
