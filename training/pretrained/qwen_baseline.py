
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading pretrained model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": "在 Linux 中，如何查看当前监听的 TCP 端口？请简洁回答。",
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][prompt_len:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("\n=== Baseline response ===")
    print(response)

    print("\n=== Metrics ===")
    print("Model:", MODEL_ID)
    print("Prompt tokens:", prompt_len)
    print("Generated tokens:", len(generated_ids))
    print("Generation time (s):", round(elapsed, 2))
    print(
        "Peak CUDA memory (GiB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    )


if __name__ == "__main__":
    main()
