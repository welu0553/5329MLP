from collections import OrderedDict
from models.module import Module
from models.layers import Linear, ReLU, Dropout, BatchNorm, GELU
from models.module import Sequential

class MLP(Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation='relu', dropout_prob=0.0, use_batchnorm=False):
        """
        :param input_dim: 输入维度
        :param hidden_dims: 隐藏层维度列表, eg. [128, 64]
        :param output_dim: 输出维度
        :param activation: 激活函数, 目前为 'relu' 和 'gelu'
        :param dropout_prob: Dropout 概率
        :param use_batchnorm: 是否在隐藏层后使用 BatchNorm
        """
        super().__init__()
        layers = OrderedDict()
        prev_dim = input_dim
        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers[f'linear{i}'] = Linear(prev_dim, hidden_dim)
            if use_batchnorm:
                layers[f'batchnorm{i}'] = BatchNorm(hidden_dim)
            # 添加激活函数层
            if activation.lower() == 'relu':
                layers[f'relu{i}'] = ReLU()
            elif activation.lower() == 'gelu':
                layers[f'gelu{i}'] = GELU()
            # 添加 Dropout 层
            if dropout_prob > 0:
                layers[f'dropout{i}'] = Dropout(dropout_prob)
            prev_dim = hidden_dim

        # 输出层
        layers['linear_out'] = Linear(prev_dim, output_dim)

        # 利用 Sequential 组合所有层
        self.model = Sequential(layers)

    def forward(self, x):
        return self.model(x)

    def backward(self, grad_output, lr=None):
        self.model.backward(grad_output, lr)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()