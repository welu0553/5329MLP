import numpy as np
from losses import Softmax
from utils.evaluation import accuracy, confusion_matrix, precision_recall_f1, macro_f1


def testing_model(model, loader):
    """
    Perform forward propagation on the test set and calculate the classification accuracy
    :param model: Model with loaded parameters
    :param test_data: Test data, shape (N, features)
    :param test_labels: Test labels, one-hot encoding, shape (N, num_classes)
    :return: Accuracy (between 0 and 1)
    """
    model.eval()

    X_test = loader.get_test_data()
    y_test_onehot = loader.get_test_labels()
    y_test = np.argmax(y_test_onehot, axis=1)

    # Forward propagation to obtain predicted logits
    logits = model(X_test)
    softmax_module = Softmax()
    probs = softmax_module.forward(logits)
    # Get the predicted category
    y_pred = np.argmax(probs, axis=1)

    # Calculate various evaluation indicators
    acc = accuracy(y_test, y_pred)
    ma_f1 = macro_f1(y_test, y_pred, 10)
    precisions, recalls, f1s = precision_recall_f1(y_test, y_pred, 10)
    cm = confusion_matrix(y_test, y_pred, 10)

    return acc, precisions, recalls, f1s, ma_f1, cm
