import torch

from training.model.embedding import TokenEmbedding
from training.model.position_embedding import PositionEmbedding


torch.manual_seed(42)

input_ids = torch.tensor([
    [1, 2, 3],
    [3, 2, 1],
])

token_embedding = TokenEmbedding(
    vocab_size=10,
    hidden_dim=8
)

position_embedding = PositionEmbedding(
    max_seq_len=16,
    hidden_dim=8
)

token_vectors = token_embedding(input_ids)
position_vectors = position_embedding(input_ids)

hidden_states = token_vectors + position_vectors

print("Token shape:", token_vectors.shape)
print("Position shape:", position_vectors.shape)
print("Combined shape:", hidden_states.shape)

assert hidden_states.shape == (2, 3, 8)

# 相同 token 出现在不同位置，
# 最终表示应当不同。
same_token_different_position = (
    hidden_states[0, 0],
    hidden_states[1, 2]
)

print(
    "Same token, different positions equal:",
    torch.allclose(*same_token_different_position)
)

assert not torch.allclose(*same_token_different_position)

print("Position Embedding test PASSED")
