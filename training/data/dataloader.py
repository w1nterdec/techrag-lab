from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size=2,
    shuffle=True
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
