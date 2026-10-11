import torch

from training.model.tiny_transformer import TinyTransformer
from training.loss.causal_lm import causal_lm_loss


torch.manual_seed(42)

model = TinyTransformer(
    vocab_size=20,
    hidden_dim=16,
    max_seq_len=32,
    num_layers=2,
)

input_ids = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
])

# 第一条样本前两个 token 的目标位置被忽略
labels = torch.tensor([
    [-100, -100, 3, 4],
    [5, 6, 7, 8],
])

logits = model(input_ids)
loss = causal_lm_loss(logits, labels)

print("Logits shape:", logits.shape)
print("Loss:", loss.item())

assert torch.isfinite(loss)

loss.backward()

all_grads = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.parameters()
)

print("All gradients valid:", all_grads)

assert all_grads

print("Causal LM loss test PASSED")
