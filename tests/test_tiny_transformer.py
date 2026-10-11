
import torch

from training.model.tiny_transformer import TinyTransformer


def test_tiny_transformer_output_shape():
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

    assert logits.shape == (2, 4, 20)


def test_tiny_transformer_causal():
    torch.manual_seed(42)

    model = TinyTransformer(
        vocab_size=20,
        hidden_dim=16,
        max_seq_len=32,
        num_layers=2,
    )
    model.eval()

    input_ids_1 = torch.tensor([[1, 2, 3, 4]])
    input_ids_2 = torch.tensor([[1, 2, 3, 9]])

    with torch.no_grad():
        logits_1 = model(input_ids_1)
        logits_2 = model(input_ids_2)

    # 改变未来 token，过去位置的预测不能变化
    torch.testing.assert_close(
        logits_1[:, :3, :],
        logits_2[:, :3, :],
        atol=1e-6,
        rtol=1e-6,
    )


def test_tiny_transformer_backward():
    torch.manual_seed(42)

    model = TinyTransformer(
        vocab_size=20,
        hidden_dim=16,
        max_seq_len=32,
        num_layers=2,
    )

    input_ids = torch.tensor([[1, 2, 3, 4]])

    logits = model(input_ids)
    loss = logits.square().mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
