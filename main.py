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
│   └── evaluation.py        # 评估指标计算
├── tests/                # 单元测试（选做）
│   └── test_models.py
└── README.md             # 项目说明文档

"""

from data_loader import DataLoader
from models import MLP
from utils import training_model, save_model, load_model, assign_parameters, testing_model
import itertools
import copy
from datetime import datetime


def training(model, para_grad, loader):
    histories = training_model(model, loader, para_grad)
    loss_history, grad_norm_history, param_norm_change_history = histories
    return loss_history, grad_norm_history, param_norm_change_history


def train_model(hyperparams, para_grad, loader):
    model = MLP(**hyperparams)
    training(model, para_grad, loader)

    # 在训练循环结束后调用保存函数
    save_paras = {
        'hyperparams': hyperparams,
        'para_grad': para_grad
    }
    model_name = './saves/' + generate_model_filename()
    save_model(model, save_paras, model_name) #  "./saves/model_test.npz"
    return save_paras, model

def test_model(loader, model_name='model_test.npz'):
    url = './saves/' + model_name
    # 读入模型
    model_params, loaded_params = load_model(url)
    hyperparams = loaded_params['hyperparams']
    print("Loaded hyperparameters:", hyperparams)
    # 利用参数恢复模型
    model = MLP(**hyperparams)
    assign_parameters(model, model_params)

    # ---------------------
    # 测试模型准确率
    result = testing_model(model, loader)
    # acc, precisions, recalls, f1s, ma_f1, cm
    [print(x, '\n') for x in result]

    save_paras = {
        'hyperparams': hyperparams,
        'para_grad': loaded_params
    }
    return save_paras, model


def permutations_params(para_grad_pool):
    """
    根据 para_grad_pool 中的各个键取值生成所有超参数组合。

    如果池中同时包含 'opt_para_SGD'、'opt_para_Adam' 和 'opt'，
    则按条件组合：当 'opt' 为 'SGD' 时，选择 'opt_para_SGD' 中的内容；
    当 'opt' 为 'Adam' 时，选择 'opt_para_Adam' 中的内容。
    否则，对所有键直接进行笛卡尔积组合。

    返回：
        一个包含所有超参数组合的列表，每个元素为一个字典。
    """
    pool = copy.deepcopy(para_grad_pool)

    # 如果池中包含特定条件键，则进行条件处理
    if all(k in pool for k in ['opt_para_SGD', 'opt_para_Adam', 'opt']):
        sgd_params = pool.pop('opt_para_SGD')
        adam_params = pool.pop('opt_para_Adam')
        base_keys = list(pool.keys())
        base_values = [pool[k] for k in base_keys]
        base_combinations = list(itertools.product(*base_values))
        permutations_params_list = []
        for comb in base_combinations:
            base_dict = dict(zip(base_keys, comb))
            if base_dict.get('opt') == 'SGD':
                for opt_param in sgd_params:
                    new_params = base_dict.copy()
                    new_params['opt_para'] = opt_param
                    permutations_params_list.append(new_params)
            elif base_dict.get('opt') == 'Adam':
                for opt_param in adam_params:
                    new_params = base_dict.copy()
                    new_params['opt_para'] = opt_param
                    permutations_params_list.append(new_params)
            else:
                # 如果 opt 值不是 'SGD' 或 'Adam'，则直接加入
                permutations_params_list.append(base_dict)
        return permutations_params_list
    else:
        # 普通参数池：对所有键直接进行笛卡尔积组合
        keys = list(pool.keys())
        values_lists = [pool[k] for k in keys]
        permutations_params_list = []
        for comb in itertools.product(*values_lists):
            permutations_params_list.append(dict(zip(keys, comb)))
        return permutations_params_list


def generate_model_filename(prefix="relu", extension="npz"):
    """
    生成一个包含当前时间戳的模型文件名，格式例如：
    "MLP_20230425_142530.npz"

    返回：
        一个字符串，作为文件名。
    """
    # 获取当前日期和时间，格式化为 YYYYMMDD_HHMMSS 的字符串
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.{extension}"
    return filename

def main():
    # 加载数据
    train_data_path = './Assignment1-Dataset/train_data.npy'
    train_label_path = './Assignment1-Dataset/train_label.npy'
    test_data_path = './Assignment1-Dataset/test_data.npy'
    test_label_path = './Assignment1-Dataset/test_label.npy'
    loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10)
    print(loader)

    # 定义参数
    hyperparams_pool = {
        'input_dim': [128],  # 请勿修改
        'hidden_dims': [[256, 128],
                        [512, 256],
                        [1024, 512, 256]],
        'output_dim': [10],  # 请勿修改
        'activation': ['relu'],  # 'gelu'
        'dropout_prob': [0.0, 0.1],
        'use_batchnorm': [True]
    }
    hyperparams_list = permutations_params(hyperparams_pool)

    para_grad_pool = {
        'learning_rate': [0.005, 0.01],
        'batch_size': [512, 256],
        'num_epochs': [200],
        'opt': ['SGD', 'Adam'],  # 'Adam'
        'opt_para_SGD': [{'momentum': 0.9, 'weight_decay': 0.0001},
                         {'momentum': 0.92, 'weight_decay': 0.0001}],
        'opt_para_Adam': [{'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-6},
                          {'beta1': 0.92, 'beta2': 0.98, 'eps': 1e-6}],
        'shuffle': [True]
    }
    para_grad_list = permutations_params(para_grad_pool)
    print(len(hyperparams_list), len(para_grad_list))
    # 训练模型
    i = 0
    for hyperparams in hyperparams_list:
        for para_grad in para_grad_list:
            i += 1
            print(f'轮数:{i}/{len(hyperparams_list) * len(para_grad_list)}')
            print('当前参数:')
            print('hyperparams:', hyperparams)
            print('para_grad:', para_grad)
            train_model(hyperparams, para_grad, loader)

    # 测试模型
    # test_model(loader, 'model_test.npz')

if __name__ == '__main__':
    main()


    '''
    # 调试用单参
    
    hyperparams = {
        'input_dim': loader.get_train_data().shape[1],
        'hidden_dims': [1024, 512, 256],
        'output_dim': 10,
        'activation': 'relu',
        'dropout_prob': 0.01,
        'use_batchnorm': True
    }
    
    para_grad = {
        'learning_rate': 0.01,
        'batch_size': 512,
        'num_epochs': 5,
        'opt': 'SGD',  # 'Adam'
        'opt_para': {'momentum': 0.9, 'weight_decay': 0.0001},
        # 'opt_para': {'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-6},
        'shuffle': True
    }
    '''