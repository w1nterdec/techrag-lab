import torch

from training.model.attention import SelfAttention


x = torch.randn(
    2,
    3,
    8
)


attention = SelfAttention(
    hidden_dim=8
)


output = attention(x)


print("Input:")
print(x.shape)


print("\nOutput:")
print(output.shape)
