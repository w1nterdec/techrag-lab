import torch

from training.model.transformer_block import TransformerBlock


torch.manual_seed(42)

model = TransformerBlock(hidden_dim=8)

x = torch.randn(
    2, 3, 8,
    requires_grad=True
)

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

loss = output.square().mean()
loss.backward()

print("Input gradient exists:", x.grad is not None)

print(
    "All parameters have gradients:",
    all(p.grad is not None for p in model.parameters())
)

assert x.grad is not None
assert all(p.grad is not None for p in model.parameters())

print("Transformer Block test PASSED")
