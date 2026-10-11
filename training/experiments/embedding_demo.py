import torch

from training.model.embedding import TokenEmbedding


input_ids = torch.tensor(
    [
        [1,2,3],
        [4,5,6]
    ]
)


embedding = TokenEmbedding(
    vocab_size=10,
    hidden_dim=8
)


output = embedding(
    input_ids
)


print("Input shape:")
print(input_ids.shape)


print("\nEmbedding output:")
print(output)


print("\nOutput shape:")
print(output.shape)
