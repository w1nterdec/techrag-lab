import torch


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path):
    return torch.load(path)
