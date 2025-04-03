import numpy as np
import time
import matplotlib.pyplot as plt

from data_loader import DataLoader
from models.mlp_model import MLP
from losses import Softmax, CrossEntropyLoss
from optimizers import SGD, Adam

# ---------------------
# 超参数设置
learning_rate = 0.1  # 可尝试不同学习率，比如 0.5、0.1、0.05 等
batch_size = 512
num_epochs = 10

# 数据路径（请根据实际情况修改）
train_data_path = './Assignment1-Dataset/train_data.npy'
train_label_path = './Assignment1-Dataset/train_label.npy'
test_data_path = './Assignment1-Dataset/test_data.npy'
test_label_path = './Assignment1-Dataset/test_label.npy'

# ---------------------
# 加载数据
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10)
print(loader)

# ---------------------
# 构建模型
input_dim = loader.get_train_data().shape[1]
hidden_dims = [128, 64, 32]
output_dim = 10
model = MLP(input_dim, hidden_dims, output_dim, activation='relu', dropout_prob=0.1, use_batchnorm=True)
print("Model built.")

# ---------------------
# 初始化损失和优化器
softmax = Softmax()
criterion = CrossEntropyLoss()
params = model.parameters()

# 使用 SGD 作为优化器（你也可以尝试 Adam）
# optimizer = SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
optimizer = Adam(params, lr=learning_rate,
                 beta1=0.9, beta2=0.999, eps=1e-8)

# ---------------------
# 用于调试的记录变量
loss_history = []
grad_norm_history = []
param_norm_change_history = []  # 记录每个 epoch 平均参数更新变化

# ---------------------
# 训练循环
for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_grad_norms = []
    param_norm_changes = []
    start_time = time.time()
    num_batches = 0

    for X_batch, y_batch in loader.batch_generator(mode='train', batch_size=batch_size, shuffle=True):
        num_batches += 1
        # 记录更新前所有参数的 L2 范数
        pre_param_norms = [np.linalg.norm(p['param']) for p in model.parameters()]

        # 前向传播
        logits = model(X_batch)
        probs = softmax.forward(logits)
        loss = criterion.forward(probs, y_batch)
        epoch_loss += loss

        # 反向传播：计算梯度并存储到各层的持久参数字典中
        grad_logits = criterion.backward(probs, y_batch)
        model.backward(grad_logits)  # 各层只计算并保存梯度

        # 记录当前 mini-batch 梯度的平均范数
        batch_norms = []
        for p in model.parameters():
            if p['grad'] is not None:
                batch_norms.append(np.linalg.norm(p['grad']))
        if batch_norms:
            batch_grad_norms.append(np.mean(batch_norms))

        # 更新参数
        optimizer.step()

        # 记录更新后所有参数的 L2 范数
        post_param_norms = [np.linalg.norm(p['param']) for p in model.parameters()]
        # 计算每个参数的更新变化（绝对值），然后求平均
        batch_change = np.mean([abs(post - pre) for pre, post in zip(pre_param_norms, post_param_norms)])
        param_norm_changes.append(batch_change)
        if batch_change < 1e-12:
            print(f"DEBUG: Epoch {epoch+1}, Batch {num_batches} param change is nearly 0")

        # 清零梯度
        optimizer.zero_grad()

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    avg_grad_norm = np.mean(batch_grad_norms) if batch_grad_norms else 0
    grad_norm_history.append(avg_grad_norm)
    avg_param_change = np.mean(param_norm_changes) if param_norm_changes else 0
    param_norm_change_history.append(avg_param_change)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.6f}, "
          f"Avg Param Change: {avg_param_change:.8f}, Time: {time.time()-start_time:.2f}s")

# 绘制训练过程中的损失、梯度范数和参数变化
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(loss_history, marker='o')
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(grad_norm_history, marker='o', color='orange')
plt.title("Avg Gradient Norm over epochs")
plt.xlabel("Epoch")
plt.ylabel("Avg Grad Norm")

plt.subplot(1, 3, 3)
plt.plot(param_norm_change_history, marker='o', color='green')
plt.title("Avg Parameter Norm Change over epochs")
plt.xlabel("Epoch")
plt.ylabel("Avg Param Change")

plt.tight_layout()
plt.show()
