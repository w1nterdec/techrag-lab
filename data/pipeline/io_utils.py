"""Validated, atomic publication of a single JSON or JSONL file.

This does not implement multi-file transactions or full power-loss durability.
"""

import json
import os
from pathlib import Path
import tempfile

if __package__:
    from .validation import validate_standard_dataset, validate_training_dataset
else:
    from validation import validate_standard_dataset, validate_training_dataset


def _atomic_write(path, write):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    descriptor = None
    stream = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None  # The stream now owns the descriptor.
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = None
        os.replace(temporary, target)
        temporary = None
    except BaseException as error:
        # Cleanup errors must not replace the original serialization/I/O error.
        if stream is not None:
            try:
                stream.close()
            except BaseException as cleanup_error:
                error.add_note(f"Temporary stream cleanup failed: {cleanup_error}")
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                error.add_note(f"Temporary descriptor cleanup failed: {cleanup_error}")
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                error.add_note(f"Temporary file cleanup failed ({temporary}): {cleanup_error}")
        raise


def write_standard_json(dataset, path):
    validate_standard_dataset(dataset)
    _atomic_write(path, lambda stream: json.dump(dataset, stream, ensure_ascii=False, indent=2))


def write_training_jsonl(dataset, path):
    validate_training_dataset(dataset)

    def write(stream):
        for item in dataset:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    _atomic_write(path, write)
