
import torch
from torch.utils.data import DataLoader

from training.data.dataset import TechRAGDataset
from training.tokenizer.tokenizer import TechRAGTokenizer
from training.collator.collator import TechRAGCollator
from training.model.tiny_transformer import TinyTransformer
from training.loss.causal_lm import causal_lm_loss


def main():
    torch.manual_seed(42)

    dataset = TechRAGDataset("data/train/train.jsonl")
    assert len(dataset) > 0, "Dataset is empty"

    tokenizer = TechRAGTokenizer()
    collator = TechRAGCollator(tokenizer)

    # 先用实际的 Prompt / Response 格式建立词表
    for sample in dataset:
        prompt, response = collator.build_prompt(sample)
        tokenizer.encode(prompt)
        tokenizer.encode(response)

    tokenizer.freeze()
    vocab_size = len(tokenizer.vocab)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    seq_len = input_ids.size(1)

    # 本实验不做截断，保证全部训练目标被保留
    max_seq_len = 64
    assert seq_len <= max_seq_len, (
        f"Sequence length {seq_len} exceeds {max_seq_len}"
    )

    # 当前模型只有 Causal Mask。
    # 本实验要求右侧 Padding，且 Padding 标签不参与 Loss。
    assert torch.all(labels[attention_mask == 0] == -100)

    assert input_ids.max().item() < vocab_size

    model = TinyTransformer(
        vocab_size=vocab_size,
        hidden_dim=32,
        max_seq_len=max_seq_len,
        num_layers=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    print("Dataset size:", len(dataset))
    print("Vocabulary size:", vocab_size)
    print("Batch shape:", tuple(input_ids.shape))
    print("Valid targets:", int((labels != -100).sum()))

    model.train()

    # 同一个小 Batch 重复训练，仅验证最小训练闭环
    for step in range(101):
        optimizer.zero_grad()

        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)

        assert torch.isfinite(loss), "Non-finite loss"

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step={step:3d}, loss={loss.item():.6f}")

    print("Real data training smoke test PASSED")


if __name__ == "__main__":
    main()
