import torch

from training.model.transformer_block import TransformerBlock


def test_transformer_block_shape():
    model = TransformerBlock(hidden_dim=8)

    x = torch.randn(2, 3, 8)

    output = model(x)

    assert output.shape == x.shape


def test_transformer_block_backward():
    model = TransformerBlock(hidden_dim=8)

    x = torch.randn(
        2, 3, 8,
        requires_grad=True
    )

    output = model(x)
    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_transformer_block_causal():
    torch.manual_seed(42)

    model = TransformerBlock(hidden_dim=8)
    model.eval()

    x1 = torch.randn(1, 4, 8)

    x2 = x1.clone()
    x2[:, 3, :] = torch.randn(1, 8) * 100

    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)

    torch.testing.assert_close(
        y1[:, :3, :],
        y2[:, :3, :],
        atol=1e-6,
        rtol=1e-6
    )
