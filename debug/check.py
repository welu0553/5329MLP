import numpy as np
from data_loader import DataLoader
from models.mlp_model import MLP
from losses import Softmax, CrossEntropyLoss

# 设置数据路径和参数（根据实际情况修改）
train_data_path = '../Assignment1-Dataset/train_data.npy'
train_label_path = '../Assignment1-Dataset/train_label.npy'
test_data_path = '../Assignment1-Dataset/test_data.npy'
test_label_path = '../Assignment1-Dataset/test_label.npy'

num_classes = 10

# 初始化数据加载器（内部会归一化数据，并将标签转换为 one-hot 编码）
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=num_classes)
print(loader)  # 打印数据集基本信息

# 从 batch_generator 中取一个 mini-batch
batch_gen = loader.batch_generator(mode='train', batch_size=32, shuffle=True)
X_batch, y_batch = next(batch_gen)
print("Mini-batch data shape:", X_batch.shape)
print("Mini-batch label shape:", y_batch.shape)

# 构建 MLP 模型
input_dim = loader.get_train_data().shape[1]
hidden_dims = [128, 64]
output_dim = num_classes
model = MLP(input_dim, hidden_dims, output_dim, activation='relu', dropout_prob=0.5, use_batchnorm=True)

# 初始化 Softmax 和交叉熵损失模块
softmax = Softmax()
criterion = CrossEntropyLoss()

# --- 前向传播 ---
logits = model(X_batch)
probs = softmax.forward(logits)
loss = criterion.forward(probs, y_batch)
print("Initial Loss: {:.4f}".format(loss))

# --- 反向传播 ---
grad_logits = criterion.backward(probs, y_batch)
# 注意：此处不传入 lr，让各层只计算梯度
model.backward(grad_logits)

# --- 检查各层梯度 ---
params = model.parameters()  # 每个参数为字典 {'param': ..., 'grad': ...}
print("\nGradient check:")
for i, p in enumerate(params):
    grad = p.get('grad', None)
    if grad is None:
        print(f"Parameter {i}: shape {p['param'].shape}, grad is None")
    else:
        norm = np.linalg.norm(grad)
        print(f"Parameter {i}: shape {p['param'].shape}, grad norm {norm:.6f}")

# 如果梯度都非常接近 0 或者异常大，你可能需要进一步检查每个层的 backward 实现，
# 或者调整学习率、权重初始化等超参数。
