import torch

from training.model.tiny_transformer import TinyTransformer
from training.loss.causal_lm import causal_lm_loss


def main():
    torch.manual_seed(42)

    model = TinyTransformer(
        vocab_size=16,
        hidden_dim=32,
        max_seq_len=16,
        num_layers=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # 人工构造的固定训练样本
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])

    # 对普通 next-token prediction，
    # labels 可以与 input_ids 相同，
    # causal_lm_loss 内部负责 shift。
    labels = input_ids.clone()

    model.train()

    for step in range(101):
        optimizer.zero_grad()

        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)

        loss.backward()

        optimizer.step()

        if step % 20 == 0:
            print(
                f"step={step:3d}, "
                f"loss={loss.item():.6f}"
            )


if __name__ == "__main__":
    main()
