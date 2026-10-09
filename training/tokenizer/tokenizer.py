class TechRAGTokenizer:

    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def encode(self, text):

        tokens = text.split()

        ids = []

        for token in tokens:

            if token not in self.vocab:

                idx = len(self.vocab) + 1

                self.vocab[token] = idx
                self.reverse_vocab[idx] = token

            ids.append(
                self.vocab[token]
            )

        return ids


    def decode(self, ids):

        return " ".join(
            self.reverse_vocab[i]
            for i in ids
        )
