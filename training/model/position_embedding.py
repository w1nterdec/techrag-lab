import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            max_seq_len,
            hidden_dim
        )

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        seq_len = input_ids.size(1)

        if seq_len > self.embedding.num_embeddings:
            raise ValueError(
                "Sequence length exceeds max_seq_len"
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        # [seq_len, hidden_dim]
        return self.embedding(positions)
