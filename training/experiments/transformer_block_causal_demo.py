import torch

from training.model.transformer_block import TransformerBlock


torch.manual_seed(42)

model = TransformerBlock(hidden_dim=8)
model.eval()

# 构造一个序列
x1 = torch.randn(1, 4, 8)

# 复制序列，只修改最后一个 token
x2 = x1.clone()
x2[:, 3, :] = torch.randn(1, 8) * 100

with torch.no_grad():
    y1 = model(x1)
    y2 = model(x2)

# 检查前 3 个 token 是否受到未来 token 的影响
past_diff = (y1[:, :3, :] - y2[:, :3, :]).abs()

print("Output shape:", y1.shape)
print("Past max difference:", past_diff.max().item())

past_unchanged = torch.allclose(
    y1[:, :3, :],
    y2[:, :3, :],
    atol=1e-6
)

last_changed = not torch.allclose(
    y1[:, 3, :],
    y2[:, 3, :],
    atol=1e-6
)

print("Past tokens unchanged:", past_unchanged)
print("Last token changed:", last_changed)

assert past_unchanged, "Future information leaked!"
assert last_changed, "Last token unexpectedly unchanged!"

print("Transformer Block causal test PASSED")
