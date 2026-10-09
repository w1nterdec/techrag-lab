from training.data.dataset import TechRAGDataset
from training.data.dataloader import create_dataloader


dataset = TechRAGDataset(
    "data/train/train.jsonl"
)


loader = create_dataloader(
    dataset,
    batch_size=2
)


batch = next(iter(loader))


print("Batch:")
print(batch)
