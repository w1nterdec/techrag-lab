import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from data.pipeline.paths import DATA_DIR, DEFAULT_PATHS, parse_stage_args


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("ingest", "clean", "dedup", "quality_filter", "build_dataset")


def fingerprints():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("raw", "cleaned", "train")
            for path in (ROOT / "data" / folder).rglob("*") if path.is_file()}


def execute(args, cwd):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", *args], cwd=cwd, env=env,
                          text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize("stage", MODULES)
def test_default_paths_are_module_relative_from_external_cwd(monkeypatch, tmp_path, stage):
    monkeypatch.chdir(tmp_path)
    source, target = parse_stage_args(stage, ())
    assert DATA_DIR == ROOT / "data"
    assert (source, target) == DEFAULT_PATHS[stage]
    assert source.is_absolute() and target.is_absolute()


@pytest.mark.parametrize("stage", MODULES)
def test_main_without_args_ignores_process_arguments(monkeypatch, tmp_path, standard_sample, stage):
    module = importlib.import_module("data.pipeline." + stage)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pytest", "--unrelated-test-argument"])
    seen = {}
    if stage == "ingest":
        def build(source, target):
            seen.update(input=source, output=target)
        monkeypatch.setattr(module, "build_dataset", build)
    else:
        def load(source):
            seen["input"] = source
            return [standard_sample]
        def save(rows, target):
            seen["output"] = target
        monkeypatch.setattr(module, "load_dataset", load)
        monkeypatch.setattr(module, "save_jsonl" if stage == "build_dataset" else "save_dataset", save)
    module.main()
    assert (seen["input"], seen["output"]) == DEFAULT_PATHS[stage]


def test_explicit_relative_paths_follow_calling_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert parse_stage_args("ingest", ["--input", "raw.json", "--output", "output/data.json"]) == (
        tmp_path / "raw.json", tmp_path / "output/data.json")


@pytest.mark.parametrize("stage", MODULES)
@pytest.mark.parametrize("mode", ["module", "script"])
@pytest.mark.parametrize("location", ["root", "external"])
def test_actual_standalone_cli_uses_only_temporary_paths(tmp_path, standard_sample, stage, mode, location):
    before = fingerprints()
    source = tmp_path / "input.json"
    target = tmp_path / "new/output"
    raw = [{"question": standard_sample["input"], "answer": standard_sample["output"], "category": "linux"}]
    source.write_text(json.dumps(raw if stage == "ingest" else [standard_sample]), encoding="utf-8")
    cwd = ROOT if location == "root" else tmp_path
    command = (["-m", "data.pipeline." + stage] if mode == "module"
               else [str(ROOT / "data/pipeline" / (stage + ".py"))])
    result = execute([*command, "--input", str(source), "--output", str(target)], cwd)
    assert result.returncode == 0, result.stderr
    if stage == "build_dataset":
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        assert rows == [{key: standard_sample[key] for key in ("instruction", "input", "output")}]
    else:
        rows = json.loads(target.read_text())
        if stage == "ingest":
            assert len(rows) == 1
            assert rows[0]["input"] == standard_sample["input"]
            assert rows[0]["output"] == standard_sample["output"]
            assert rows[0]["metadata"]["category"] == "linux"
        else:
            assert rows == [standard_sample]
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]
    assert fingerprints() == before


@pytest.mark.parametrize("location", ["root", "external"])
def test_real_unified_module_cli_content_and_report(tmp_path, location):
    before = fingerprints()
    source = tmp_path / "raw.json"
    raw = [
        {"question": "如何查看 Linux 监听端口？", "answer": "短"},
        {"question": "如何查看 Linux 监听端口？", "answer": "可以使用 ss -tlnp 命令查看系统监听端口。"},
        {"question": "如何查看 Linux 监听端口？", "answer": "另一个合格答案应被首条合格答案去重。"},
        {"question": "  如何查看 Docker 容器列表？  ", "answer": "  可以使用 docker ps 命令查看运行中的容器。  "},
    ]
    source.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output"
    result = execute(["-m", "data.pipeline.run_pipeline", "--input", str(source),
                      "--output-dir", str(output)], ROOT if location == "root" else tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    expected = [{"instruction": "Answer the technical question based on the provided information.",
                 "input": raw[index]["question"].strip(), "output": raw[index]["answer"].strip()}
                for index in (1, 3)]
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert rows == expected
    assert all(set(row) == {"instruction", "input", "output"} for row in rows)
    assert [row["output_count"] for row in report["stages"]] == [4, 4, 3, 2, 2, 2]
    assert report["status"] == "completed" and report["output_committed"] is True
    assert sorted(p.name for p in output.iterdir()) == ["train.jsonl"]
    assert fingerprints() == before


def test_unified_cli_requires_explicit_output_directory(tmp_path):
    before = fingerprints()
    result = execute(["-m", "data.pipeline.run_pipeline"], tmp_path)
    assert result.returncode == 2
    assert "--output-dir" in result.stderr
    assert list(tmp_path.iterdir()) == []
    assert fingerprints() == before


def test_unified_cli_default_input_is_module_relative(tmp_path):
    before = fingerprints()
    output = tmp_path / "output"
    result = execute(["-m", "data.pipeline.run_pipeline", "--output-dir", str(output)], tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["input"] == str(ROOT / "data/raw/sample.json")
    raw = json.loads((ROOT / "data/raw/sample.json").read_text())
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert rows == [{"instruction": "Answer the technical question based on the provided information.",
                     "input": item["question"], "output": item["answer"]} for item in raw]
    assert fingerprints() == before


def test_unified_cli_returns_failure_report(tmp_path):
    before = fingerprints()
    source = tmp_path / "raw.json"
    source.write_text('[{"question":"q","answer":false}]')
    output = tmp_path / "output"
    result = execute(["-m", "data.pipeline.run_pipeline", "--input", str(source),
                      "--output-dir", str(output)], tmp_path)
    assert result.returncode == 1
    assert result.stdout == ""
    report = json.loads(result.stderr)
    assert report["failed_stage"] == "ingest"
    assert report["error"]["type"] == "DatasetValidationError"
    assert (report["error"]["index"], report["error"]["field"], report["error"]["layer"]) == (0, "answer", "raw")
    assert not output.exists()
    assert fingerprints() == before
