import numpy as np
import time

from data_loader import DataLoader
from models.mlp_model import MLP
from losses import Softmax, CrossEntropyLoss
from optimizers import SGD  # 或者使用 Adam

# 超参数设置
learning_rate = 0.05
batch_size = 32
num_epochs = 10

# 数据路径（请根据实际路径调整）
train_data_path = '../Assignment1-Dataset/train_data.npy'
train_label_path = '../Assignment1-Dataset/train_label.npy'
test_data_path = '../Assignment1-Dataset/test_data.npy'
test_label_path = '../Assignment1-Dataset/test_label.npy'

# 加载数据
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10)
print(loader)

# 构建模型
# 假设输入特征数由训练数据 shape 得到
input_dim = loader.get_train_data().shape[1]
hidden_dims = [128, 64]  # 示例：两层隐藏层
output_dim = 10  # 10 分类问题

model = MLP(input_dim, hidden_dims, output_dim, activation='relu', dropout_prob=0.5, use_batchnorm=True)

# 初始化损失函数与 softmax 模块
softmax = Softmax()
criterion = CrossEntropyLoss()

# 收集模型参数（要求每个参数返回格式为 {'param': ..., 'grad': ...}）
params = model.parameters()

# 初始化优化器（使用 SGD，这里也可以换成 Adam）
optimizer = SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)

# 训练循环
for epoch in range(num_epochs):
    epoch_loss = 0.0
    start_time = time.time()
    # 使用 mini-batch 生成器，mode 设置为 'train'，并打乱数据
    for X_batch, y_batch in loader.batch_generator(mode='train', batch_size=batch_size, shuffle=True):
        # 前向传播：得到 logits
        logits = model(X_batch)
        # 计算 softmax 概率分布
        probs = softmax.forward(logits)
        # 计算交叉熵损失（注意 y_batch 已经是 one-hot 编码）
        loss = criterion.forward(probs, y_batch)
        epoch_loss += loss

        # 反向传播：先计算交叉熵的梯度，再传递给模型计算各层梯度
        grad_logits = criterion.backward(probs, y_batch)
        # 模型 backward 只计算梯度（内部各层将梯度存储到 grad 属性中）
        model.backward(grad_logits)

        # 更新参数：由优化器根据各层保存的梯度更新参数
        optimizer.step()
        # 清零梯度，为下一次迭代做准备
        optimizer.zero_grad()

    avg_loss = epoch_loss / (loader.get_train_data().shape[0] / batch_size)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")
