import torch.nn as nn

from training.model.embedding import TokenEmbedding
from training.model.position_embedding import PositionEmbedding
from training.model.transformer_block import TransformerBlock


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=64,
        max_seq_len=128,
        num_layers=2,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if hidden_dim <= 0 or max_seq_len <= 0 or num_layers <= 0:
            raise ValueError("Model dimensions must be positive")

        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )

        self.position_embedding = PositionEmbedding(
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=False,
        )

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        if input_ids.size(1) > self.max_seq_len:
            raise ValueError("Input exceeds max_seq_len")

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(input_ids)
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits
