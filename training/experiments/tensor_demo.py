from training.data.dataset import TechRAGDataset
from training.tokenizer.tokenizer import TechRAGTokenizer
from training.collator.collator import TechRAGCollator


dataset = TechRAGDataset(
    "data/train/train.jsonl"
)


tokenizer = TechRAGTokenizer()


collator = TechRAGCollator(
    tokenizer
)


batch = collator(
    [
        dataset[0],
        dataset[1]
    ]
)


print(batch)


print("\ninput_ids:")
print(batch["input_ids"])


print("\nlabels:")
print(batch["labels"])


print("\nattention_mask:")
print(batch["attention_mask"])
