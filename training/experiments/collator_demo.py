from training.data.dataset import TechRAGDataset
from training.collator.collator import TechRAGCollator


dataset = TechRAGDataset(
    "data/train/train.jsonl"
)


collator = TechRAGCollator()


result = collator(
    [
        dataset[0],
        dataset[1]
    ]
)


print(result)
