
import torch


class TechRAGTokenizer:
    def __init__(self):
        # 特殊 Token
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        self.reverse_vocab = {
            0: "<PAD>",
            1: "<UNK>",
        }

        # False: 允许扩充词表
        # True: 词表冻结，未知 Token 映射到 <UNK>
        self.frozen = False

    def encode(self, text):
        tokens = text.split()
        ids = []

        for token in tokens:
            if token not in self.vocab:
                if self.frozen:
                    ids.append(self.vocab["<UNK>"])
                    continue

                idx = len(self.vocab)
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token

            ids.append(self.vocab[token])

        return ids

    def batch_encode(self, texts):
        encoded = [
            self.encode(text)
            for text in texts
        ]

        if not encoded:
            raise ValueError("texts must not be empty")

        max_len = max(len(ids) for ids in encoded)

        input_ids = []
        attention_mask = []

        for ids in encoded:
            padding_length = max_len - len(ids)

            input_ids.append(
                ids + [self.vocab["<PAD>"]] * padding_length
            )

            attention_mask.append(
                [1] * len(ids) + [0] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }

    def decode(self, ids):
        return " ".join(
            self.reverse_vocab[i]
            for i in ids
            if i != self.vocab["<PAD>"]
        )

    def freeze(self):
        """冻结词表，阻止新 Token 被添加。"""
        self.frozen = True
