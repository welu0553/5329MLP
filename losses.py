import numpy as np

class Softmax:
    def __init__(self):
        self.probs = None

    def forward(self, logits):
        """
        计算 SoftMax 概率分布, 进行数值稳定性处理
        :param logits: 模型输出, 形状（N，C）
        :return:概率分布，形状（N，C）
        """
        # 数值稳定性处理
        exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad_output=None):
        """
        单独的 softmax 不需要独立计算反向传播，
        在需要反向传播时通常与交叉熵一起融合计算梯度。
        注：可以额外实现 Jacobian 矩阵计算。
        """
        raise NotImplementedError("Softmax backward is usually combined with loss backward.")

class CrossEntropyLoss:
    def forward(self, probs, labels):
        """
        交叉上损失计算
        :param probs: SoftMax 输出的概率分布，形状（N，C）
        :param labels: one-hot 编码的标签，形状（N，C）
        :return: 平均损失（标量）
        """
        N = probs.shape[0]
        loss = -np.sum(labels * np.log(probs + 1e-8)) / N
        return loss

    def backward(self, probs, labels):
        '''计算交叉熵损失关于 logits 的梯度'''
        N = probs.shape[0]
        grad = (probs - labels) / N
        return grad