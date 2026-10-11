import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        hidden_dim
    ):
        super().__init__()


        self.q_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        self.scale = hidden_dim ** 0.5



    def forward(self, x):

        # x:
        # (batch, seq_len, hidden_dim)

        Q = self.q_proj(x)

        K = self.k_proj(x)

        V = self.v_proj(x)


        # QK^T
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )


	 # Scaling
        scores = scores / self.scale

        # Causal Mask
        seq_len = x.size(1)

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            )
        )

        # Mask future positions BEFORE softmax
        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf")
        )

        # Attention weights
        weights = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted sum
        output = torch.matmul(
            weights,
            V
        )

        return output
