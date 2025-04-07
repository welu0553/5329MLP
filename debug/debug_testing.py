import numpy as np
from data_loader import DataLoader
from models.mlp_model import MLP
from utils import load_model, testing_model, assign_parameters


# 导入数据
train_data_path = '../Assignment1-Dataset/train_data.npy'
train_label_path = '../Assignment1-Dataset/train_label.npy'
test_data_path = '../Assignment1-Dataset/test_data.npy'
test_label_path = '../Assignment1-Dataset/test_label.npy'
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10)
# 读入模型
model_params, loaded_params = load_model("../saves/model_test.npz")
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


