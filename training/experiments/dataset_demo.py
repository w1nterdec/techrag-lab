from training.data.dataset import TechRAGDataset


dataset = TechRAGDataset(
    "data/train/train.jsonl"
)


print("Dataset size:", len(dataset))


print("\nFirst sample:")

print(dataset[0])
