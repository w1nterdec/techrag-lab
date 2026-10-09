import torch
from torch import nn

# 1. 固定随机种子
torch.manual_seed(42)

# 2. 选择 GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 3. 构造训练数据：y = 3x + 2
x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
y = 3 * x + 2

# 4. 创建线性模型
model = nn.Linear(1, 1).to(device)

print("Device:", device)
print("Initial weight:", model.weight.item())
print("Initial bias:", model.bias.item())

# 5. 定义损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# 6. 开始训练
for epoch in range(200):
    prediction = model(x)
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {loss.item():.6f}"
        )

# 7. 查看训练结果
print("\nTraining completed!")
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())
