import pytest
import torch
import torch.nn.functional as F

from training.loss.causal_lm import causal_lm_loss


def test_loss_matches_manual_shift():
    torch.manual_seed(42)

    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ])

    actual = causal_lm_loss(logits, labels)

    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 10),
        labels[:, 1:].reshape(-1),
    )

    torch.testing.assert_close(actual, expected)


def test_ignored_logits_do_not_affect_loss():
    torch.manual_seed(42)

    logits1 = torch.randn(1, 4, 10)
    logits2 = logits1.clone()

    labels = torch.tensor([[-100, -100, 3, 4]])

    # logits[:, 0, :] 对应 labels[:, 1]
    # 该目标是 -100，因此应该被忽略
    logits2[:, 0, :] = torch.randn(10) * 100

    loss1 = causal_lm_loss(logits1, labels)
    loss2 = causal_lm_loss(logits2, labels)

    torch.testing.assert_close(loss1, loss2)


def test_all_targets_ignored_raises():
    logits = torch.randn(1, 4, 10)
    labels = torch.full((1, 4), -100)

    with pytest.raises(ValueError, match="No valid target"):
        causal_lm_loss(logits, labels)


def test_short_sequence_raises():
    logits = torch.randn(1, 1, 10)
    labels = torch.tensor([[2]])

    with pytest.raises(ValueError, match="seq_len"):
        causal_lm_loss(logits, labels)
