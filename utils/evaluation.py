import numpy as np

def accuracy(y_true, y_pred):
    """
    计算准确率
    :param y_true: 真实标签，一维 numpy 数组
    :param y_pred: 预测标签，一维 numpy 数组
    :return: 准确率（float）
    """
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, num_classes):
    """
    计算混淆矩阵
    :param y_true: 真实标签，一维 numpy 数组
    :param y_pred: 预测标签，一维 numpy 数组
    :param num_classes: 类别数
    :return: 混淆矩阵，二维 numpy 数组，形状 (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1(y_true, y_pred, num_classes):
    """
    分别计算每个类别的精确率、召回率和 F1 分数
    :param y_true: 真实标签，一维 numpy 数组
    :param y_pred: 预测标签，一维 numpy 数组
    :param num_classes: 类别数
    :return: (precisions, recalls, f1s) 三个 numpy 数组，每个长度为 num_classes
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    f1s = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1s[i] = (2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
                  if (precisions[i] + recalls[i]) > 0 else 0)
    return precisions, recalls, f1s

def macro_f1(y_true, y_pred, num_classes):
    """
    计算宏平均 F1 分数
    :param y_true: 真实标签，一维 numpy 数组
    :param y_pred: 预测标签，一维 numpy 数组
    :param num_classes: 类别数
    :return: 宏 F1 分数（float）
    """
    _, _, f1s = precision_recall_f1(y_true, y_pred, num_classes)
    return np.mean(f1s)
