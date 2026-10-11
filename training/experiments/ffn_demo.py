import torch

from training.model.feed_forward import FeedForward


torch.manual_seed(42)

ffn = FeedForward(hidden_dim=8)

x = torch.randn(2, 3, 8, requires_grad=True)

output = ffn(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

loss = output.square().mean()
loss.backward()

print("Input gradient exists:", x.grad is not None)
print(
    "All FFN parameters have gradients:",
    all(p.grad is not None for p in ffn.parameters())
)
