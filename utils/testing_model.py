import numpy as np
from losses import Softmax
from utils.evaluation import accuracy, confusion_matrix, precision_recall_f1, macro_f1

def testing_model(model, loader):
    """
    对测试集进行前向传播，计算分类准确率
    :param model: 已加载参数的模型
    :param test_data: 测试数据，形状 (N, features)
    :param test_labels: 测试标签，one-hot 编码，形状 (N, num_classes)
    :return: 准确率（0~1之间）
    """
    model.eval()

    X_test = loader.get_test_data()
    y_test_onehot = loader.get_test_labels()
    y_test = np.argmax(y_test_onehot, axis=1)

    # 前向传播获取预测 logits
    logits = model(X_test)
    softmax_module = Softmax()
    probs = softmax_module.forward(logits)
    # 得到预测的类别
    y_pred = np.argmax(probs, axis=1)

    # 计算各种评估指标
    acc = accuracy(y_test, y_pred)
    ma_f1 = macro_f1(y_test, y_pred, 10)
    precisions, recalls, f1s = precision_recall_f1(y_test, y_pred, 10)
    cm = confusion_matrix(y_test, y_pred, 10)

    return acc, precisions, recalls, f1s, ma_f1, cm
