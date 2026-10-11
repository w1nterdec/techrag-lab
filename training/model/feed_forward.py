import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()

        intermediate_dim = hidden_dim * expansion_factor

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)
