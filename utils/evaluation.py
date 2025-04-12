import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculate accuracy
    :param y_true: true label, one-dimensional numpy array
    :param y_pred: predicted label, one-dimensional numpy array
    :return: accuracy (float)
    """
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Calculate confusion matrix
    :param y_true: true label, 1D numpy array
    :param y_pred: predicted label, 1D numpy array
    :param num_classes: number of classes
    :return: confusion matrix, 2D numpy array, shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1(y_true, y_pred, num_classes):
    """
    Calculate the precision, recall and F1 score for each category
    :param y_true: true label, one-dimensional numpy array
    :param y_pred: predicted label, one-dimensional numpy array
    :param num_classes: number of categories
    :return: (precisions, recalls, f1s) three numpy arrays, each with a length of num_classes
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
    Calculate the macro average F1 score
    :param y_true: true label, 1D numpy array
    :param y_pred: predicted label, 1D numpy array
    :param num_classes: number of classes
    :return: macro F1 score (float)
    """
    _, _, f1s = precision_recall_f1(y_true, y_pred, num_classes)
    return np.mean(f1s)
