
import torch
from torch.utils.data import DataLoader

from training.data.dataset import TechRAGDataset
from training.tokenizer.tokenizer import TechRAGTokenizer
from training.collator.collator import TechRAGCollator
from training.model.tiny_transformer import TinyTransformer
from training.loss.causal_lm import causal_lm_loss


def test_training_data_integration():
    dataset = TechRAGDataset("data/train/train.jsonl")

    assert len(dataset) > 0

    tokenizer = TechRAGTokenizer()
    collator = TechRAGCollator(tokenizer)

    # 1. 先用与训练一致的模板建立词表
    for sample in dataset:
        prompt, response = collator.build_prompt(sample)
        tokenizer.encode(prompt)
        tokenizer.encode(response)

    tokenizer.freeze()
    vocab_size = len(tokenizer.vocab)

    # 2. 构建真正的 DataLoader Batch
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

    # 3. 检查张量契约
    assert input_ids.shape == labels.shape
    assert input_ids.shape == attention_mask.shape
    assert len(tokenizer.vocab) == vocab_size

    assert input_ids.min().item() >= 0
    assert input_ids.max().item() < vocab_size

    valid_labels = labels[labels != -100]
    assert valid_labels.numel() > 0
    assert valid_labels.min().item() >= 0
    assert valid_labels.max().item() < vocab_size

    # PAD 不得作为训练目标
    assert torch.all(labels[attention_mask == 0] == -100)

    # 4. 集成到真实模型和 Loss
    torch.manual_seed(42)

    model = TinyTransformer(
        vocab_size=vocab_size,
        hidden_dim=32,
        max_seq_len=64,
        num_layers=2,
    )

    assert input_ids.size(1) <= model.max_seq_len

    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)

    assert logits.shape == (
        input_ids.size(0),
        input_ids.size(1),
        vocab_size,
    )
    assert torch.isfinite(loss)

    # 5. 验证反向传播
    loss.backward()

    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
