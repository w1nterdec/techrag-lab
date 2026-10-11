import torch

from training.model.attention import SelfAttention


torch.manual_seed(42)

attention = SelfAttention(hidden_dim=8)
attention.eval()

# 同一个序列
x1 = torch.randn(1, 4, 8)
x2 = x1.clone()

# 仅修改最后一个 token
x2[:, 3, :] = torch.randn(1, 8) * 100

with torch.no_grad():
    y1 = attention(x1)
    y2 = attention(x2)

print("Output shape:", y1.shape)

# 前三个位置不应该受到第四个 token 的影响
past_unchanged = torch.allclose(
    y1[:, :3, :],
    y2[:, :3, :],
    atol=1e-6
)

print("Past tokens unchanged:", past_unchanged)

# 第四个位置允许发生变化
last_changed = not torch.allclose(
    y1[:, 3, :],
    y2[:, 3, :],
    atol=1e-6
)

print("Last token changed:", last_changed)

assert past_unchanged, "Future information leaked!"
assert last_changed, "Last token unexpectedly unchanged!"

print("Causal attention test PASSED")
