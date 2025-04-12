# 5329MLP
### 1. 项目说明

This project implements a modular MLP (Multi-Layer Perceptron) neural network training platform based on Python and Numpy, supporting the entire process from data loading, model building, batch training, automatic evaluation, model selection, to ablation experiments. The core functions of this project are divided into the following parts:

Data import and processing Supports data files in .npyformat as input; supports loading of training and test sets; automatically handles basic preprocessing operations such as data normalisation and label conversion; uses a custom DataLoader class to encapsulate data batch processing logic, and supports the setting of parameters such as batch_size and shuffle.
Model structure construction The model is implemented based on a custom Module base class and Sequential container; supports user-defined MLP network layers, the dimension of each layer (hidden_dims), the type of activation function, whether to use BatchNorm, whether to use Dropout, etc.; supports the following network components: fully connected layer (Linear) activation function: ReLU, GELU (implemented using an approximation function) Dropout layer Batch Normalization layer
Model training module automatically traverses all combinations of hyperparameters for batch training; supports SGD and Adam optimizers, and also supports parameter adjustment such as momentum, weight decay, beta1, beta2, eps, etc.; each model is automatically saved in .npzformat after training is complete; supports unified configuration of training parameters such as epochs, learning rate, batch size, optimizer, etc.
Model testing and evaluation automatically reads the saved model file and uniformly restores and reconstructs the model; The evaluation function in evaluation.pyis used to calculate the following metrics: Accuracy macro-F1 The evaluation results of all models are saved as a all_model_results.csvfile, which contains the model file name, accuracy, F1 score, hyperparameter information, etc.
Model selection and result output Automatically select the model with the highest macro-F1 score among all models as the ‘best model’; The accuracy, F1 score and corresponding hyperparameter configuration of the best model are output; the visualization is saved as an image best_model_result.png.
Ablation experiment Based on the structure of the ‘best model’, the following three structural ablation experiments are performed: Remove Dropout (set to 0) Turn off BatchNorm Replace the activation function from ReLU to GELU Each structural variant is reinitialised and trained to ensure the fairness of the experiment; After all variants are tested, the ablation comparison chart ablation_result.pngis automatically generated to show the changes in Accuracy and Macro-F1; The specific hyperparameter configuration of each ablation experiment is also output for easy tracking and reproduction.

### 2. 项目架构
```plaintext
project_root/
├── main.py               # 程序入口。可以通过设置超参数进行训练或测试模型
├── data_loader.py        # 数据加载模块。将传入的数据输出为可迭代对象，并可选择性使用mini-batch及onehot编码
├── losses.py             # 损失函数。包含Softmax和CrossEntropy
├── optimizers.py         # 优化器。包含SGD, Adam
├── models/               # 模型模块
│   ├── __init__.py       # Module initialization
│   ├── module.py         # 基础模型模块。包含所有层的父类Module以及序列类Sequential
│   ├── layers.py         # Implementation of layers. 包含Linear, ReLU, GELU, Dropout, BatchNorm
│   └── mlp_model.py      # MLP模块。用于实现MLP的各项功能
├── utils/                # Utility functions
│   ├── __init__.py       # Module initialization
│   ├── evaluation.py     # 评估指标
│   ├── functions.py      # 函数文件。包含各种用到的函数，增强代码可读性
│   ├── training_model.py # 模型训练脚本文件。封装了整个模型训练脚本，便于在主函数中训练模型
│   └── testing_model.py  # 模型测试脚本文件。封装了整个模型测试脚本，便于在主函数中测试模型
└── README.md             # Project documentation
```