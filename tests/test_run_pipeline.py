import copy
import hashlib
import json
from pathlib import Path

import pytest

from data.pipeline import build_dataset, clean, dedup, ingest, quality_filter, run_pipeline as runner
from data.pipeline.validation import DatasetValidationError


ROOT = Path(__file__).resolve().parents[1]


def formal_fingerprints():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("raw", "cleaned", "train")
            for path in sorted((ROOT / "data" / folder).rglob("*")) if path.is_file()}


@pytest.fixture
def raw_file(tmp_path):
    raw = [
        {"question": "如何查看 Linux 监听端口？", "answer": "短"},
        {"question": "如何查看 Linux 监听端口？", "answer": "可以使用 ss -tlnp 命令查看系统监听端口。"},
        {"question": "如何查看 Linux 监听端口？", "answer": "也可以使用 ss -ltn 命令查看监听端口。"},
        {"question": "  如何查看 Docker 容器列表？  ", "answer": "  可以使用 docker ps 命令查看运行中的容器。  "},
        {"question": "", "answer": "这个答案长度足够，但问题为空。"},
        {"question": "Hi", "answer": "这个答案长度足够，但问题过短。"},
    ]
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def test_real_pipeline_data_flow_report_and_formal_data_safety(monkeypatch, tmp_path, raw_file):
    before = formal_fingerprints()
    calls = []
    snapshots = {}
    load_calls = []
    original_load = ingest.load_raw_data

    def load(path):
        load_calls.append(path)
        return original_load(path)

    monkeypatch.setattr(ingest, "load_raw_data", load)

    def wrap(module, attribute, name):
        original = getattr(module, attribute)

        def traced(data):
            calls.append(name)
            snapshots[name] = {"input": copy.deepcopy(data)}
            result = original(data)
            snapshots[name]["output"] = copy.deepcopy(result)
            return result

        monkeypatch.setattr(module, attribute, traced)

    wrap(ingest, "ingest_records", "ingest")
    wrap(clean, "clean_dataset", "clean")
    wrap(dedup, "deduplicate", "dedup")
    wrap(build_dataset, "build_training_dataset", "training")
    original_quality = quality_filter.quality_filter
    quality_calls = 0

    def quality(data):
        nonlocal quality_calls
        name = "quality_pre_filter" if quality_calls == 0 else "quality_recheck"
        quality_calls += 1
        calls.append(name)
        snapshots[name] = {"input": copy.deepcopy(data)}
        result = original_quality(data)
        snapshots[name]["output"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(quality_filter, "quality_filter", quality)
    output_dir = tmp_path / "output"
    report = runner.run_pipeline(raw_file, output_dir)
    assert calls == list(runner.STAGE_NAMES)
    assert load_calls == [raw_file]
    assert quality_calls == 2
    assert snapshots["ingest"]["input"] == json.loads(raw_file.read_text())
    for previous, following in zip(runner.STAGE_NAMES, runner.STAGE_NAMES[1:]):
        assert snapshots[previous]["output"] == snapshots[following]["input"]
    assert [row["id"] for row in snapshots["clean"]["output"]] == [
        "techrag_00000", "techrag_00001", "techrag_00002", "techrag_00003", "techrag_00005"]
    assert [row["id"] for row in snapshots["quality_pre_filter"]["output"]] == [
        "techrag_00001", "techrag_00002", "techrag_00003"]
    assert [row["id"] for row in snapshots["dedup"]["output"]] == ["techrag_00001", "techrag_00003"]
    expected = [
        {"instruction": "Answer the technical question based on the provided information.",
         "input": "如何查看 Linux 监听端口？", "output": "可以使用 ss -tlnp 命令查看系统监听端口。"},
        {"instruction": "Answer the technical question based on the provided information.",
         "input": "如何查看 Docker 容器列表？", "output": "可以使用 docker ps 命令查看运行中的容器。"},
    ]
    actual = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
    assert actual == snapshots["training"]["output"] == expected
    assert all(set(row) == {"instruction", "input", "output"} for row in actual)
    assert sorted(p.name for p in output_dir.iterdir()) == ["train.jsonl"]
    assert report["status"] == "completed" and report["output_committed"] is True
    assert report["failed_stage"] is report["error"] is None
    assert [row["input_count"] for row in report["stages"]] == [6, 6, 5, 3, 2, 2]
    assert [row["output_count"] for row in report["stages"]] == [6, 5, 3, 2, 2, 2]
    assert all(row["status"] == "completed" and row["elapsed_seconds"] >= 0
               for row in report["stages"])
    assert report["elapsed_seconds"] >= 0
    assert formal_fingerprints() == before


@pytest.mark.parametrize("failed_stage", runner.STAGE_NAMES)
def test_stage_failure_stops_flow_and_preserves_cause(monkeypatch, tmp_path, raw_file, failed_stage):
    original = RuntimeError("injected " + failed_stage + " failure")
    visited = []

    def wrap(module, attribute, stage):
        actual = getattr(module, attribute)

        def action(data):
            visited.append(stage)
            if stage == failed_stage:
                raise original
            return actual(data)

        monkeypatch.setattr(module, attribute, action)

    wrap(ingest, "ingest_records", "ingest")
    wrap(clean, "clean_dataset", "clean")
    wrap(dedup, "deduplicate", "dedup")
    wrap(build_dataset, "build_training_dataset", "training")
    actual_quality = quality_filter.quality_filter
    calls = 0

    def quality(data):
        nonlocal calls
        stage = "quality_pre_filter" if calls == 0 else "quality_recheck"
        calls += 1
        visited.append(stage)
        if stage == failed_stage:
            raise original
        return actual_quality(data)

    monkeypatch.setattr(quality_filter, "quality_filter", quality)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    target = output_dir / "train.jsonl"
    target.write_bytes(b"old")
    with pytest.raises(runner.PipelineExecutionError) as caught:
        runner.run_pipeline(raw_file, output_dir)
    assert caught.value.__cause__ is original
    assert caught.value.stage == failed_stage
    report = caught.value.report
    index = runner.STAGE_NAMES.index(failed_stage)
    assert visited == list(runner.STAGE_NAMES[:index + 1])
    assert [row["status"] for row in report["stages"]] == (
        ["completed"] * index + ["failed"] + ["not_run"] * (5 - index))
    assert report["error"] == {"type": "RuntimeError", "message": str(original)}
    assert report["status"] == "failed" and report["output_committed"] is False
    assert report["elapsed_seconds"] >= 0
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in output_dir.iterdir()) == ["train.jsonl"]


def test_dataset_error_keeps_field_attributes_and_cause(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text('[{"question":"q","answer":"a"},{"question":"q","answer":null}]')
    with pytest.raises(runner.PipelineExecutionError) as caught:
        runner.run_pipeline(path, tmp_path / "output")
    original = caught.value.__cause__
    assert isinstance(original, DatasetValidationError)
    assert (original.index, original.field, original.layer) == (1, "answer", "raw")
    assert caught.value.report["error"] == {
        "type": "DatasetValidationError", "message": str(original),
        "index": 1, "field": "answer", "layer": "raw"}
    assert caught.value.report["stages"][0]["operation"] == "convert"
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("input_kind,exception_type", [("missing", "FileNotFoundError"), ("malformed", "JSONDecodeError")])
def test_loading_failure_is_reported(tmp_path, input_kind, exception_type):
    path = tmp_path / "raw.json"
    if input_kind == "malformed":
        path.write_text("{")
    with pytest.raises(runner.PipelineExecutionError) as caught:
        runner.run_pipeline(path, tmp_path / "output")
    report = caught.value.report
    assert report["failed_stage"] == "ingest"
    assert report["error"]["type"] == exception_type
    assert report["stages"][0]["operation"] == "load"
    assert not (tmp_path / "output").exists()


def test_final_replace_failure_is_reported_without_publication(monkeypatch, tmp_path, raw_file):
    from data.pipeline import io_utils
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    target = output_dir / "train.jsonl"
    target.write_bytes(b"old")
    original = OSError("injected final replace failure")

    def fail(*args):
        raise original

    monkeypatch.setattr(io_utils.os, "replace", fail)
    with pytest.raises(runner.PipelineExecutionError) as caught:
        runner.run_pipeline(raw_file, output_dir)
    assert caught.value.__cause__ is original
    report = caught.value.report
    assert report["failed_stage"] == "training"
    assert report["stages"][-1]["operation"] == "save"
    assert report["stages"][-1]["built_count"] == 2
    assert report["output_committed"] is False
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in output_dir.iterdir()) == ["train.jsonl"]


def test_empty_pipeline_output(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text("[]")
    output_dir = tmp_path / "output"
    report = runner.run_pipeline(source, output_dir)
    assert (output_dir / "train.jsonl").read_bytes() == b""
    assert all(row["input_count"] == row["output_count"] == 0 for row in report["stages"])
    assert report["output_committed"] is True
