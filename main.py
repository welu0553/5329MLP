"""
流程:
    1. 数据集导入模块

    2. MLP模型模块

    3. 损失函数优化器模块

    4. 训练 / 测试模块

project_root/
├── main.py               # 程序入口，整合训练/测试流程
├── config.py             # 配置文件，存放超参数、路径等信息
├── data_loader.py        # 数据加载模块（DataLoader 类）
├── losses.py             # 损失函数模块（Softmax, CrossEntropy 等）
├── optimizers.py         # 优化器模块（SGD、Adam 等）
├── models/               # 模型相关模块
│   ├── __init__.py       # 模块初始化文件
│   ├── module.py         # Module 基类及 Sequential 容器
│   ├── layers.py         # 各种层的实现（Linear, ReLU, GELU, Dropout, BatchNorm 等）
│   └── mlp_model.py      # MLP 模型定义，组合各个基础层
├── utils/                # 工具模块（如评估指标、绘图函数等，可选）
│   └── metrics.py        # 评估指标计算
├── tests/                # 单元测试（选做）
│   └── test_models.py
└── README.md             # 项目说明文档

"""
import numpy as np
import data_loader
import models
import losses
import optimizers
def main():
    dataloader = data_loader.DataLoader(
        train_data_path='./Assignment1-Dataset/train_data.npy',
        train_label_path='./Assignment1-Dataset/train_label.npy',
        test_data_path='./Assignment1-Dataset/test_data.npy',
        test_label_path='./Assignment1-Dataset/test_label.npy'
    )
    print(dataloader)


    train_data = dataloader.train_data
    train_label = dataloader.train_label
    print("Train data shape:", train_data.shape, "min:", np.min(train_data), "max:", np.max(train_data))
    print("Train label shape:", train_label.shape, "unique labels:", np.unique(train_label))

if __name__ == '__main__':
    main()