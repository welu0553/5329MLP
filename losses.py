import numpy as np

class Softmax:
    def __init__(self):
        # To store softmax probabilities for backward use or inspection
        self.probs = None

    def forward(self, logits):
        """
        Compute the softmax probabilities for a batch of logits.

        Args:
            logits (ndarray): Input logits of shape (N, C), where N is the number of samples and C is the number of classes.

        Returns:
            ndarray: Softmax probability distributions of shape (N, C).
        """
        # For numerical stability, subtract max from logits before exponentiating
        exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad_output=None):
        """
        Softmax backward is generally not used independently.
        Instead, it is usually combined with CrossEntropyLoss to optimize efficiency and numerical stability.

        Raises:
            NotImplementedError: This method is intentionally not implemented.
        """
        raise NotImplementedError("Softmax backward is usually combined with loss backward.")


class CrossEntropyLoss:
    def forward(self, probs, labels):
        """
        Compute the average cross-entropy loss between predicted probabilities and one-hot labels.

        Args:
            probs (ndarray): Predicted softmax probabilities of shape (N, C).
            labels (ndarray): One-hot encoded ground truth labels of shape (N, C).

        Returns:
            float: Mean cross-entropy loss over the batch.
        """
        N = probs.shape[0]
        # Add small epsilon to prevent log(0)
        loss = -np.sum(labels * np.log(probs + 1e-8)) / N
        return loss

    def backward(self, probs, labels):
        """
        Compute the gradient of the cross-entropy loss with respect to logits
        assuming softmax was applied beforehand.

        Args:
            probs (ndarray): Predicted softmax probabilities (N, C).
            labels (ndarray): One-hot encoded ground truth (N, C).

        Returns:
            ndarray: Gradient of the loss with respect to logits.
        """
        N = probs.shape[0]
        grad = (probs - labels) / N
        return grad
