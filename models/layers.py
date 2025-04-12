import numpy as np
from models.module import Module

class Linear(Module):
    """
    Fully-connected linear layer: performs affine transformation y = xW + b.

    Attributes:
        W (np.ndarray): Weight matrix of shape (in_features, out_features), initialized using He initialization.
        b (np.ndarray): Bias vector of shape (1, out_features), initialized to zeros.
        grad_W (np.ndarray): Gradient of weights computed during backpropagation.
        grad_b (np.ndarray): Gradient of biases computed during backpropagation.
        _param_dict (list): List of dictionaries to store parameters and their gradients for optimizer access.
    """
    def __init__(self, in_features, out_features, weight_decay=0.0):
        super().__init__()
        self.weight_decay = weight_decay
        self.W = np.random.randn(in_features, out_features).astype(np.float64) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features), dtype=np.float64)
        self.grad_W = None
        self.grad_b = None
        self._param_dict = [{'param': self.W, 'grad': None},
                            {'param': self.b, 'grad': None}]

    def forward(self, x):
        self.x = np.atleast_2d(x)
        return np.dot(self.x, self.W) + self.b

    def backward(self, d_out, lr=None):
        d_out = np.atleast_2d(d_out)
        # Compute gradients with optional L2 regularization (weight decay)
        self.grad_W = np.dot(self.x.T, d_out) + self.weight_decay * self.W
        self.grad_b = np.sum(d_out, axis=0, keepdims=True)
        dX = np.dot(d_out, self.W.T)
        # If learning rate is provided (for non-optimizer use), update weights directly
        if lr is not None:
            self.W -= lr * self.grad_W
            self.b -= lr * self.grad_b
        self._param_dict[0]['grad'] = self.grad_W
        self._param_dict[1]['grad'] = self.grad_b
        return dX

    def parameters(self):
        return self._param_dict


class ReLU(Module):
    """
    ReLU activation function: f(x) = max(0, x)

    Attributes:
        x (np.ndarray): Input to be used during backpropagation.
    """
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        # Zero out gradients for inputs where the original input was <= 0
        grad_input[self.x <= 0] = 0
        return grad_input

    def parameters(self):
        return []


class GELU(Module):
    """
    GELU (Gaussian Error Linear Unit) activation using approximate formula.

    Attributes:
        x (np.ndarray): Input tensor.
        out (np.ndarray): Output after applying GELU activation.
    """
    def forward(self, x):
        self.x = x
        self.out = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))))
        return self.out

    def parameters(self):
        return []


class Dropout(Module):
    """
    Dropout layer for regularization during training.

    Attributes:
        dropout_prob (float): Probability of dropping a unit (default: 0.5)
        mask (np.ndarray): Binary mask applied to inputs during training.
    """
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.mask = None

    def forward(self, x):
        if self.training:
            # Generate dropout mask: keep probability (1 - dropout_prob)
            # Scale the retained values to maintain expectation of input
            self.mask = (np.random.rand(*x.shape) > self.dropout_prob) / (1 - self.dropout_prob)
            return x * self.mask
        else:
            return x

    def parameters(self):
        return []

    def backward(self, grad_output):
        if self.training:
            # Use same dropout mask as forward to ensure consistency
            return grad_output * self.mask
        else:
            return grad_output


class BatchNorm(Module):
    """
    Batch Normalization layer for stabilizing training.

    Attributes:
        gamma (np.ndarray): Learnable scaling parameter.
        beta (np.ndarray): Learnable shifting parameter.
        running_mean (np.ndarray): Running average of feature-wise mean.
        running_var (np.ndarray): Running average of feature-wise variance.
        grad_gamma (np.ndarray): Gradient of gamma.
        grad_beta (np.ndarray): Gradient of beta.
        _param_dict (list): List of dictionaries holding parameters and their gradients.
    """
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.gamma = np.ones((1, num_features), dtype=np.float64)
        self.beta = np.zeros((1, num_features), dtype=np.float64)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((1, num_features), dtype=np.float64)
        self.running_var = np.zeros((1, num_features), dtype=np.float64)
        self.grad_gamma = None
        self.grad_beta = None
        self._param_dict = [{'param': self.gamma, 'grad': None},
                            {'param': self.beta, 'grad': None}]

    def forward(self, x):
        if self.training:
            # Compute batch mean and variance
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            self.x_centered = x - self.mean
            self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
            self.x_norm = self.x_centered * self.std_inv
            # Update running statistics for inference use
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            out = self.gamma * self.x_norm + self.beta
        else:
            # In inference mode, use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, grad_output):
        N, D = grad_output.shape
        # Compute gradients for scale and shift
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        dxhat = grad_output * self.gamma
        # Backpropagate through normalization
        dx = (1.0 / N) * self.std_inv * (
            N * dxhat - np.sum(dxhat, axis=0, keepdims=True) -
            self.x_norm * np.sum(dxhat * self.x_norm, axis=0, keepdims=True)
        )
        self._param_dict[0]['grad'] = self.grad_gamma
        self._param_dict[1]['grad'] = self.grad_beta
        return dx

    def parameters(self):
        return self._param_dict
