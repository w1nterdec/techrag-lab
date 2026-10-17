
"""Run a reproducible Qwen baseline on a fixed QA dataset."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation/baselines/linux_qa_v1.json"
OUTPUT_DIR = ROOT / "outputs/baselines"

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def load_dataset(path):
    raw = path.read_bytes()
    items = json.loads(raw)

    if not isinstance(items, list) or not items:
        raise ValueError("Dataset must be a non-empty JSON array")

    seen_ids = set()

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every sample must be an object")

        sample_id = item.get("id")
        question = item.get("question")

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Every sample needs a non-empty id")

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question: {sample_id}")

        seen_ids.add(sample_id)

    digest = hashlib.sha256(raw).hexdigest()
    return items, digest


def get_git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_baseline(dataset_path, max_new_tokens):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline")

    questions, dataset_hash = load_dataset(dataset_path)

    device = "cuda:0"
    torch.manual_seed(42)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    print("Loading pretrained model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    model_revision = getattr(
        model.config, "_commit_hash", None
    )

    tokenizer_revision = tokenizer.init_kwargs.get(
        "_commit_hash"
    )

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "enable_thinking": False,
    }

    results = []

    for index, item in enumerate(questions, start=1):
        sample_id = item["id"]
        question = item["question"]

        print(f"\n[{index}/{len(questions)}] {sample_id}")

        messages = [
            {"role": "user", "content": question}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_ids = output[0][prompt_tokens:]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Detect why generation stopped.
        eos_ids = model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id

        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        eos_ids = set(eos_ids or [])

        ended_with_eos = (
            len(generated_ids) > 0
            and generated_ids[-1].item() in eos_ids
        )

        hit_length_limit = (
            len(generated_ids) >= max_new_tokens
        )

        if ended_with_eos:
            finish_reason = "eos"
        elif hit_length_limit:
            finish_reason = "length"
        else:
            finish_reason = "other"

        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        print(response)
        print(f"Latency: {elapsed:.2f}s")

        results.append({
            "id": sample_id,
            "category": item.get("category"),
            "question": question,
            "response": response,
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(len(generated_ids)),
            "ended_with_eos": bool(ended_with_eos),
            "hit_length_limit": bool(hit_length_limit),
            "finish_reason": finish_reason,
            "latency_seconds": round(elapsed, 4),
            "peak_cuda_allocated_gib": round(peak_memory, 4),
        })

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")

    report = {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "model": {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "dtype": "bfloat16",
        },
        "dataset": {
            "path": str(dataset_path.relative_to(ROOT)),
            "sha256": dataset_hash,
            "sample_count": len(questions),
        },
        "environment": {
            "git_commit": get_git_commit(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "generation": generation_config,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{run_id}.json"

    # Exclusive creation prevents accidental overwrite.
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nSaved baseline: {output_path}")
    print(f"Completed: {len(results)} questions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()

    if not dataset_path.is_relative_to(ROOT):
        raise ValueError("Dataset must be inside the repository")

    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    run_baseline(dataset_path, args.max_new_tokens)


if __name__ == "__main__":
    main()
