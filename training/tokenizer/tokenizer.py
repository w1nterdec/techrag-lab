import torch


class TechRAGTokenizer:

    def __init__(self):

        self.vocab = {
            "<PAD>": 0
        }

        self.reverse_vocab = {
            0: "<PAD>"
        }


    def encode(self, text):

        tokens = text.split()

        ids = []

        for token in tokens:

            if token not in self.vocab:

                idx = len(self.vocab)

                self.vocab[token] = idx
                self.reverse_vocab[idx] = token

            ids.append(
                self.vocab[token]
            )

        return ids


    def batch_encode(
        self,
        texts
    ):

        encoded = [
            self.encode(text)
            for text in texts
        ]


        max_len = max(
            len(x)
            for x in encoded
        )


        input_ids = []
        attention_mask = []


        for ids in encoded:

            padding_length = (
                max_len - len(ids)
            )


            input_ids.append(
                ids + [0] * padding_length
            )


            attention_mask.append(
                [1] * len(ids)
                +
                [0] * padding_length
            )


        return {
            "input_ids": torch.tensor(
                input_ids
            ),

            "attention_mask": torch.tensor(
                attention_mask
            )
        }


    def decode(self, ids):

        return " ".join(
            self.reverse_vocab[i]
            for i in ids
            if i != 0
        )
