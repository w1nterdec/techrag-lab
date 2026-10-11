import torch.nn as nn

from training.model.attention import SelfAttention
from training.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = SelfAttention(hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim)

    def forward(self, x):
        # Attention + Residual
        x = x + self.attention(self.norm1(x))

        # FFN + Residual
        x = x + self.ffn(self.norm2(x))

        return x
