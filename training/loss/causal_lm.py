import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]

    labels == -100 的位置忽略。
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")

    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T]")

    if logits.size(1) < 2:
        raise ValueError("seq_len must be at least 2")

    # 第 t 个位置预测第 t+1 个 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid = shift_labels.ne(-100)

    # 避免全为 -100 时出现 NaN
    if not valid.any().item():
        raise ValueError("No valid target tokens in labels")

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    return loss
