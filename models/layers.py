import numpy as np
from models.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, weight_decay=0.0):
        super().__init__()
        self.weight_decay = weight_decay
        # 建议使用正态分布的 He 初始化
        # self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        # self.b = np.zeros((1, out_features))
        self.W = np.random.randn(in_features, out_features).astype(np.float64) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features), dtype=np.float64)

        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = np.atleast_2d(x)
        return np.dot(self.x, self.W) + self.b

    def backward(self, d_out, lr=None):
        d_out = np.atleast_2d(d_out)
        self.grad_W = np.dot(self.x.T, d_out) + self.weight_decay * self.W
        self.grad_b = np.sum(d_out, axis=0, keepdims=True)
        dX = np.dot(d_out, self.W.T)
        # 如果 lr 非空，则直接更新（训练时传入 lr=None，让优化器更新）
        if lr is not None:
            self.W -= lr * self.grad_W
            self.b -= lr * self.grad_b
        return dX

    def parameters(self):
        # 返回字典列表，便于优化器统一处理
        return [{'param': self.W, 'grad': self.grad_W},
                {'param': self.b, 'grad': self.grad_b}]

class ReLU(Module):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        # 屏蔽负值
        grad_input[self.x <= 0] = 0
        return grad_input

    def parameters(self):
        return []  # 无参数


'''额外实现'''
class GELU(Module):
    def forward(self, x):
        self.x = x
        # 采用近似公式
        self.out = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))))
        return self.out

    def parameters(self):
        return []  # 无参数

class Dropout(Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.mask = None

    def forward(self, x):
        if self.training:
            # 生成 mask，并进行反向缩放，保证期望不变
            """
            mask解释：
                (np.random.rand(*x.shape) > self.dropout_prob)：
                    仅保留大于概率的点 <==> 去掉不满足概率的点
                / (1 - self.dropout_prob)：
                    缩放，保证放缩前后期望不变
                    explain:
                        原:
                            E = (1 - dropout_prob) * a + dropout_prob * 0
                              = (1 - dropout_prob) * a
                        -> 整体期望减小
                        After 放缩: 
                            E = (1 - dropout_prob) * a / (1 - dropout_prob) + dropout_prob * 0
                              = a
                        -> 整体期望不变
            """
            self.mask = (np.random.rand(*x.shape) > self.dropout_prob) / (1 - self.dropout_prob)
            return x * self.mask
        else:
            return x

    def parameters(self):
        return []  # 无参数

    def backward(self, grad_output):
        if self.training:
            # 使用相同的 dropout 掩码，用来make sure前向和反向传播的一致性
            return grad_output * self.mask
        else:
            return grad_output

class BatchNorm(Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        # self.gamma = np.ones((1, num_features))
        # self.beta = np.zeros((1, num_features))
        self.gamma = np.ones((1, num_features), dtype=np.float64)
        self.beta = np.zeros((1, num_features), dtype=np.float64)

        self.momentum = momentum
        self.eps = eps
        # self.running_mean = np.zeros((1, num_features))
        # self.running_var = np.zeros((1, num_features))
        self.running_mean = np.zeros((1, num_features), dtype=np.float64)
        self.running_var = np.zeros((1, num_features), dtype=np.float64)
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x):
        if self.training:
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            self.x_centered = x - self.mean
            self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
            self.x_norm = self.x_centered * self.std_inv
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            out = self.gamma * self.x_norm + self.beta
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, grad_output):
        N, D = grad_output.shape
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        dxhat = grad_output * self.gamma
        dx = (1.0 / N) * self.std_inv * (
            N * dxhat - np.sum(dxhat, axis=0, keepdims=True) -
            self.x_norm * np.sum(dxhat * self.x_norm, axis=0, keepdims=True)
        )
        return dx

    def parameters(self):
        return [{'param': self.gamma, 'grad': self.grad_gamma},
                {'param': self.beta, 'grad': self.grad_beta}]
