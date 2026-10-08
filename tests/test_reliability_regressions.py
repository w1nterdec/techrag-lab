import json
from pathlib import Path

import pytest

from data.pipeline import build_dataset, clean, dedup, ingest, quality_filter


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_dedup_default_paths_are_independent_of_cwd(monkeypatch, tmp_path, standard_sample):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def load(path):
        seen["input"] = path
        return [standard_sample]

    def save(rows, path):
        seen["output"] = path
        assert rows == [standard_sample]

    monkeypatch.setattr(dedup, "load_dataset", load)
    monkeypatch.setattr(dedup, "save_dataset", save)
    dedup.main()
    assert seen == {"input": DATA_DIR / "cleaned/clean_dataset.json",
                    "output": DATA_DIR / "cleaned/dedup_dataset.json"}


@pytest.mark.parametrize("save", [clean.save_dataset, dedup.save_dataset,
                                  quality_filter.save_dataset, build_dataset.save_jsonl])
def test_partial_serialization_failure_preserves_old_target(monkeypatch, tmp_path,
                                                            standard_sample, save):
    target = tmp_path / "output"
    target.write_bytes(b"old-complete-output")
    failure = RuntimeError("injected partial serialization failure")

    if save is build_dataset.save_jsonl:
        original = json.dumps
        calls = 0

        def fail_dumps(item, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise failure
            return original(item, **kwargs)

        monkeypatch.setattr(json, "dumps", fail_dumps)
    else:
        def fail_dump(rows, file, **kwargs):
            file.write("[")
            raise failure

        monkeypatch.setattr(json, "dump", fail_dump)

    with pytest.raises(RuntimeError) as caught:
        save([standard_sample, standard_sample], target)
    assert caught.value is failure
    assert target.read_bytes() == b"old-complete-output"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output"]


def test_ingest_partial_write_preserves_old_target(monkeypatch, tmp_path):
    target = tmp_path / "output"
    target.write_bytes(b"old-complete-output")
    monkeypatch.setattr(ingest, "load_raw_data", lambda path: [{"question": "q", "answer": "a"}])

    def fail_dump(rows, file, **kwargs):
        file.write("[")
        raise RuntimeError("injected ingest serialization failure")

    monkeypatch.setattr(json, "dump", fail_dump)
    with pytest.raises(RuntimeError):
        ingest.build_dataset("unused-input", target)
    assert target.read_bytes() == b"old-complete-output"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output"]
