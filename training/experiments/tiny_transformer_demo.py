import torch

from training.model.tiny_transformer import TinyTransformer


torch.manual_seed(42)

model = TinyTransformer(
    vocab_size=20,
    hidden_dim=16,
    max_seq_len=32,
    num_layers=2,
)

input_ids = torch.tensor([
    [1, 2, 3, 4],
    [4, 3, 2, 1],
])

logits = model(input_ids)

print("Input shape:", input_ids.shape)
print("Logits shape:", logits.shape)

assert logits.shape == (2, 4, 20)

loss = logits.square().mean()
loss.backward()

all_grads = all(
    parameter.grad is not None
    for parameter in model.parameters()
)

print("All parameters have gradients:", all_grads)

assert all_grads

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("Parameter count:", parameter_count)
print("Tiny Transformer forward/backward PASSED")
