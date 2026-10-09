import json
from pathlib import Path

from torch.utils.data import Dataset


class TechRAGDataset(Dataset):

    def __init__(self, path):

        self.path = Path(path)

        self.samples = []

        with open(
            self.path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if line.strip():

                    self.samples.append(
                        json.loads(line)
                    )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        return self.samples[index]
