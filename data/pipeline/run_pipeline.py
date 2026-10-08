"""Run the dataset pipeline in memory and publish only train.jsonl."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from . import build_dataset, clean, dedup, ingest, quality_filter
from .paths import DATA_DIR
from .validation import DatasetValidationError


STAGE_NAMES = ("ingest", "clean", "quality_pre_filter", "dedup", "quality_recheck", "training")


class PipelineExecutionError(RuntimeError):
    def __init__(self, stage, report):
        self.stage = stage
        self.report = report
        error = report["error"]
        super().__init__(f"{stage}: {error['type']}: {error['message']}")


def _error_details(error):
    details = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, DatasetValidationError):
        details.update(index=error.index, field=error.field, layer=error.layer)
    return details


def run_pipeline(input_path, output_dir):
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve() / "train.jsonl"
    started = perf_counter()
    report = {
        "status": "running", "input": str(input_path), "output": str(output_path),
        "output_committed": False, "failed_stage": None, "error": None,
        "elapsed_seconds": 0.0,
        "stages": [
            {"name": name, "status": "not_run", "input_count": None,
             "output_count": None, "elapsed_seconds": 0.0, "operation": None}
            for name in STAGE_NAMES
        ],
    }
    stages = {row["name"]: row for row in report["stages"]}

    def execute(name, input_count, operation, action):
        row = stages[name]
        row.update(status="running", input_count=input_count, operation=operation)
        stage_started = perf_counter()
        try:
            result = action(row)
        except Exception as error:
            row.update(status="failed", elapsed_seconds=perf_counter() - stage_started)
            details = _error_details(error)
            row["error"] = details
            report.update(status="failed", failed_stage=name, error=details,
                          elapsed_seconds=perf_counter() - started)
            raise PipelineExecutionError(name, report) from error
        row.update(status="completed", output_count=len(result),
                   elapsed_seconds=perf_counter() - stage_started)
        return result

    def ingest_action(row):
        raw = ingest.load_raw_data(input_path)
        row["input_count"] = len(raw) if isinstance(raw, list) else None
        row["operation"] = "convert"
        return ingest.ingest_records(raw)

    dataset = execute("ingest", None, "load", ingest_action)
    dataset = execute("clean", len(dataset), "clean", lambda row: clean.clean_dataset(dataset))
    dataset = execute("quality_pre_filter", len(dataset), "filter",
                      lambda row: quality_filter.quality_filter(dataset))
    dataset = execute("dedup", len(dataset), "deduplicate", lambda row: dedup.deduplicate(dataset))
    dataset = execute("quality_recheck", len(dataset), "filter",
                      lambda row: quality_filter.quality_filter(dataset))

    def training_action(row):
        training = build_dataset.build_training_dataset(dataset)
        row["operation"] = "save"
        row["built_count"] = len(training)
        build_dataset.save_jsonl(training, output_path)
        report["output_committed"] = True
        return training

    execute("training", len(dataset), "build", training_action)
    report.update(status="completed", elapsed_seconds=perf_counter() - started)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the TechRAG-Lab dataset pipeline")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "raw/sample.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_pipeline(args.input, args.output_dir)
    except PipelineExecutionError as error:
        print(json.dumps(error.report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
