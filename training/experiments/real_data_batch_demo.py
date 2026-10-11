
from torch.utils.data import DataLoader

from training.data.dataset import TechRAGDataset
from training.tokenizer.tokenizer import TechRAGTokenizer
from training.collator.collator import TechRAGCollator


def main():
    dataset = TechRAGDataset("data/train/train.jsonl")

    assert len(dataset) > 0, "Dataset is empty"

    tokenizer = TechRAGTokenizer()
    collator = TechRAGCollator(tokenizer)

    # 第一遍：按照实际训练格式建立词表
    for sample in dataset:
        prompt, response = collator.build_prompt(sample)

        tokenizer.encode(prompt)
        tokenizer.encode(response)

    tokenizer.freeze()
    vocab_size = len(tokenizer.vocab)

    print("Dataset size:", len(dataset))
    print("Vocabulary size:", vocab_size)
    print("Tokenizer frozen:", tokenizer.frozen)

    # 第二遍：通过 DataLoader + Collator 生成 Batch
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print("Input IDs shape:", input_ids.shape)
    print("Attention mask shape:", attention_mask.shape)
    print("Labels shape:", labels.shape)

    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape == labels.shape

    # 冻结后词表不能增长
    assert len(tokenizer.vocab) == vocab_size

    # 所有输入 ID 必须处于模型词表范围
    assert input_ids.min().item() >= 0
    assert input_ids.max().item() < vocab_size

    # 有效标签必须处于词表范围
    valid_labels = labels[labels != -100]

    assert valid_labels.numel() > 0
    assert valid_labels.min().item() >= 0
    assert valid_labels.max().item() < vocab_size

    # Padding 位置不得参与 Loss
    assert (labels[attention_mask == 0] == -100).all()

    print("Valid target tokens:", valid_labels.numel())
    print("Real data batch test PASSED")


if __name__ == "__main__":
    main()
