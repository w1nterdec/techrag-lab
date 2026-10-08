import json
import os
from pathlib import Path

import pytest

from data.pipeline import io_utils
from data.pipeline.validation import DatasetValidationError


WRITERS = [io_utils.write_standard_json, io_utils.write_training_jsonl]


class FaultStream:
    def __init__(self, stream, fault):
        self.stream = stream
        self.fault = fault
        self.close_calls = 0

    def write(self, text):
        if self.fault == "write":
            self.stream.write(text[:1])
            raise OSError("injected write failure")
        return self.stream.write(text)

    def flush(self):
        if self.fault == "flush":
            raise OSError("injected flush failure")
        return self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        self.close_calls += 1
        if self.fault == "close" and self.close_calls == 1:
            raise OSError("injected close failure")
        return self.stream.close()


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("fault", ["mkdir", "create", "fdopen", "write", "flush", "fsync", "close", "replace"])
def test_io_failure_preserves_target_and_cleans_temp(monkeypatch, tmp_path,
                                                     standard_sample, writer, exists, fault):
    target = tmp_path / "target"
    if exists:
        target.write_bytes(b"old-complete-output")
    descriptors = []
    original_create = io_utils.tempfile.mkstemp
    original_fdopen = io_utils.os.fdopen

    def create(**kwargs):
        fd, name = original_create(**kwargs)
        descriptors.append(fd)
        assert Path(name).parent == target.parent
        return fd, name

    def fail(*args, **kwargs):
        raise OSError("injected " + fault + " failure")

    monkeypatch.setattr(io_utils.tempfile, "mkstemp", create)
    if fault == "mkdir":
        monkeypatch.setattr(Path, "mkdir", fail)
    elif fault == "create":
        monkeypatch.setattr(io_utils.tempfile, "mkstemp", fail)
    elif fault == "fdopen":
        monkeypatch.setattr(io_utils.os, "fdopen", fail)
    elif fault in ("write", "flush", "close"):
        monkeypatch.setattr(io_utils.os, "fdopen",
                            lambda *args, **kwargs: FaultStream(original_fdopen(*args, **kwargs), fault))
    elif fault == "fsync":
        monkeypatch.setattr(io_utils.os, "fsync", fail)
    elif fault == "replace":
        monkeypatch.setattr(io_utils.os, "replace", fail)

    with pytest.raises(OSError, match="injected"):
        writer([standard_sample], target)
    if exists:
        assert target.read_bytes() == b"old-complete-output"
    else:
        assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (["target"] if exists else [])
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("writer", WRITERS)
def test_structure_failure_creates_no_output_directory(monkeypatch, tmp_path, standard_sample, writer):
    target = tmp_path / "new/sub/target"
    calls = []

    def forbidden_create(**kwargs):
        calls.append(kwargs)
        raise AssertionError("temporary created before validation")

    monkeypatch.setattr(io_utils.tempfile, "mkstemp", forbidden_create)
    bad = dict(standard_sample, output=None)
    with pytest.raises(DatasetValidationError) as caught:
        writer([standard_sample, bad], target)
    assert (caught.value.index, caught.value.field) == (1, "output")
    assert calls == []
    assert not target.parent.exists()


@pytest.mark.parametrize("writer", WRITERS)
def test_unserializable_extra_only_damages_temporary_file(tmp_path, standard_sample, writer):
    target = tmp_path / "target"
    target.write_bytes(b"old")
    standard_sample["extra"] = object()
    with pytest.raises(TypeError):
        writer([standard_sample], target)
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["target"]


@pytest.mark.parametrize("writer", WRITERS)
def test_cleanup_failure_does_not_mask_original(monkeypatch, tmp_path, standard_sample, writer):
    target = tmp_path / "target"
    target.write_bytes(b"old")
    original = OSError("original replace failure")
    original_unlink = Path.unlink

    def fail_replace(*args):
        raise original

    def fail_unlink(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise PermissionError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(io_utils.os, "replace", fail_replace)
        patch.setattr(Path, "unlink", fail_unlink)
        with pytest.raises(OSError) as caught:
            writer([standard_sample], target)
        assert caught.value is original
        assert any("cleanup failure" in note for note in caught.value.__notes__)
        assert target.read_bytes() == b"old"
    residues = list(tmp_path.glob("*.tmp"))
    assert len(residues) == 1
    for path in residues:
        path.unlink()


def test_stream_cleanup_failure_preserves_serialization_exception(monkeypatch, tmp_path, standard_sample):
    original = RuntimeError("original serialization failure")
    original_fdopen = io_utils.os.fdopen

    class CloseFailure(FaultStream):
        def close(self):
            self.stream.close()
            raise OSError("injected stream cleanup failure")

    def fail_dump(*args, **kwargs):
        raise original

    monkeypatch.setattr(io_utils.os, "fdopen",
                        lambda *args, **kwargs: CloseFailure(original_fdopen(*args, **kwargs), None))
    monkeypatch.setattr(io_utils.json, "dump", fail_dump)
    with pytest.raises(RuntimeError) as caught:
        io_utils.write_standard_json([standard_sample], tmp_path / "target")
    assert caught.value is original
    assert any("stream cleanup failure" in note for note in original.__notes__)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("writer", WRITERS)
def test_success_closes_and_syncs_before_replace(monkeypatch, tmp_path, standard_sample, writer):
    target = tmp_path / "target"
    target.write_bytes(b"old")
    events = []
    original_fdopen = io_utils.os.fdopen
    original_fsync = io_utils.os.fsync
    original_replace = io_utils.os.replace
    streams = []

    class TrackedStream(FaultStream):
        def flush(self):
            events.append("flush")
            return self.stream.flush()

        def close(self):
            events.append("close")
            return self.stream.close()

    def fdopen(*args, **kwargs):
        stream = TrackedStream(original_fdopen(*args, **kwargs), None)
        streams.append(stream)
        return stream

    def fsync(fd):
        events.append("fsync")
        original_fsync(fd)

    def replace(source, destination):
        assert Path(source).parent == target.parent
        assert target.read_bytes() == b"old"
        assert streams[0].stream.closed
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(io_utils.os, "fdopen", fdopen)
    monkeypatch.setattr(io_utils.os, "fsync", fsync)
    monkeypatch.setattr(io_utils.os, "replace", replace)
    writer([standard_sample], target)
    assert events == ["flush", "fsync", "close", "replace"]
    expected = (json.dumps([standard_sample], ensure_ascii=False, indent=2)
                if writer is io_utils.write_standard_json
                else json.dumps(standard_sample, ensure_ascii=False) + "\n")
    assert target.read_text(encoding="utf-8") == expected
    assert sorted(p.name for p in tmp_path.iterdir()) == ["target"]
