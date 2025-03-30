import numpy as np
import time
import matplotlib.pyplot as plt

from data_loader import DataLoader
from models.mlp_model import MLP
from losses import Softmax, CrossEntropyLoss
from optimizers import SGD

def check_project():
    # ---------------------
    # 数据路径（请根据实际情况修改）
    train_data_path = '../Assignment1-Dataset/train_data.npy'
    train_label_path = '../Assignment1-Dataset/train_label.npy'
    test_data_path = '../Assignment1-Dataset/test_data.npy'
    test_label_path = '../Assignment1-Dataset/test_label.npy'
    num_classes = 10

    # ---------------------
    # 加载数据
    print("Loading data ...")
    loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=num_classes)
    print(loader)

    # 从 mini-batch 生成器中获取一个批次
    batch_gen = loader.batch_generator(mode='train', batch_size=32, shuffle=True)
    X_batch, y_batch = next(batch_gen)
    print(f"Mini-batch shapes: X: {X_batch.shape}, y: {y_batch.shape}")

    # ---------------------
    # 构建模型
    input_dim = loader.get_train_data().shape[1]
    hidden_dims = [128, 64]
    output_dim = num_classes
    model = MLP(input_dim, hidden_dims, output_dim, activation='relu', dropout_prob=0.5, use_batchnorm=True)
    print("Model built.")

    # ---------------------
    # 初始化损失函数与 Softmax 模块
    softmax = Softmax()
    criterion = CrossEntropyLoss()

    # ---------------------
    # 前向传播
    logits = model(X_batch)
    probs = softmax.forward(logits)
    loss = criterion.forward(probs, y_batch)
    print(f"Initial Loss: {loss:.4f}")

    # ---------------------
    # 反向传播，计算梯度
    grad_logits = criterion.backward(probs, y_batch)
    model.backward(grad_logits)  # 各层只计算并保存梯度，不更新参数

    # 检查各层梯度范数
    params = model.parameters()  # 每个参数为字典 {'param': ..., 'grad': ...}
    print("\nGradient check:")
    for i, p in enumerate(params):
        grad = p.get('grad', None)
        if grad is None:
            print(f"Parameter {i}: shape {p['param'].shape}, grad is None")
        else:
            norm = np.linalg.norm(grad)
            print(f"Parameter {i}: shape {p['param'].shape}, grad norm: {norm:.6f}")

    # ---------------------
    # 参数更新前记录参数范数
    pre_param_norms = [np.linalg.norm(p['param']) for p in params]

    # 创建优化器，例如 SGD
    optimizer = SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)

    # 执行一次参数更新
    optimizer.step()
    optimizer.zero_grad()

    # 更新后记录参数范数
    post_param_norms = [np.linalg.norm(p['param']) for p in params]
    print("\nParameter norm changes after one optimizer step:")
    for i, (pre, post) in enumerate(zip(pre_param_norms, post_param_norms)):
        print(f"Parameter {i}: pre-norm = {pre:.6f}, post-norm = {post:.6f}, change = {post - pre:.6f}")

    # ---------------------
    # 可选：绘制参数和梯度分布图（例如直方图）
    plt.figure(figsize=(12, 4))
    for i, p in enumerate(params):
        if p['grad'] is not None:
            plt.subplot(1, len(params), i+1)
            plt.hist(p['grad'].flatten(), bins=20)
            plt.title(f'Param {i} grad hist')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    check_project()
