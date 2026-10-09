import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from training.utils.checkpoint import save_checkpoint, load_checkpoint


# -----------------------------
# 1. 准备数据
# y = 3x + 2
# -----------------------------

x_train = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0]],
)

y_train = torch.tensor(
    [[5.0], [8.0], [11.0], [14.0]],
)


# -----------------------------
# 2. 创建模型
# -----------------------------

def create_model():
    return nn.Linear(1, 1)


# -----------------------------
# 3. 训练函数
# -----------------------------

def train(model, optimizer, epochs):

    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        prediction = model(x_train)

        loss = loss_fn(
            prediction,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | Loss: {loss.item():.6f}"
            )

    return loss.item()


# -----------------------------
# 第一阶段训练
# -----------------------------

print("\n===== Stage 1 Training =====")

model = create_model()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)


loss = train(
    model,
    optimizer,
    50
)


# -----------------------------
# 保存 checkpoint
# -----------------------------

checkpoint_path = Path(
    "training/experiments/checkpoint.pt"
)


state = {

    "epoch": 50,

    "loss": loss,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict()

}


save_checkpoint(
    state,
    checkpoint_path
)


print("\nCheckpoint saved.")


# -----------------------------
# 模拟程序中断
# -----------------------------

print("\n===== Simulate Restart =====")


model2 = create_model()

optimizer2 = optim.SGD(
    model2.parameters(),
    lr=0.01
)


# -----------------------------
# 加载状态
# -----------------------------

checkpoint = load_checkpoint(
    checkpoint_path
)


model2.load_state_dict(
    checkpoint["model_state_dict"]
)


optimizer2.load_state_dict(
    checkpoint["optimizer_state_dict"]
)


print(
    "Loaded epoch:",
    checkpoint["epoch"]
)

print(
    "Loaded loss:",
    checkpoint["loss"]
)


# -----------------------------
# 继续训练
# -----------------------------

print("\n===== Continue Training =====")


train(
    model2,
    optimizer2,
    50
)


print("\nTraining finished.")


print(
    "\nFinal weight:",
    model2.weight.item()
)

print(
    "Final bias:",
    model2.bias.item()
)
