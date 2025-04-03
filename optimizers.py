import numpy as np

class Optimizer:
    '''Optimizer 基类'''
    def __init__(self, params, lr=0.01):
        """
        :param params: 参数列表，每个元素为字典格式 {'param': numpy array, 'grad': numpy array}
        :param lr: 学习率
        """
        self.params = params
        self.lr = lr

    def step(self):
        '''执行一次参数更新'''
        raise NotImplementedError

    def zero_grad(self):
        '''参数清零'''
        for p in self.params:
            p['grad'] = np.zeros_like(p['grad'])

    # def zero_grad(self):
    #     for p in self.params:
    #         if p['grad'] is not None:
    #             p['grad'] = np.zeros_like(p['param'])

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        # 为每个参数创建一个形状相同的速率数组
        # self.velocities = [np.zeros_like(p['param']) for p in self.params]
        self.velocities = [np.zeros_like(p['param'], dtype=np.float64) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p['grad'] is None:
                continue
            # 确保 p['param'] 是 numpy 数组且 dtype 为 float64
            p['param'] = np.asarray(p['param'], dtype=np.float64)
            # 计算梯度时加入权重衰减
            grad = p['grad'] + self.weight_decay * p['param']
            # print(p['grad']) # 一直为0！！！！！！！
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            self.velocities[i] = np.asarray(self.velocities[i], dtype=np.float64)
            # 更新参数
            # p['param'] = p['param'] + self.velocities[i]
            # print(f"type = {type(self.velocities[i])}, dtype = {self.velocities[i].dtype}")
            p['param'][:] += self.velocities[i]  # 直接原地修改 p['param']，确保更新作用到原数组上


class Adam(Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-6):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p['param'], dtype=np.float64) for p in self.params]
        self.v = [np.zeros_like(p['param'], dtype=np.float64) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p['grad'] is None:
                continue
            grad = p['grad']
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            # 调试打印信息：
            # print(f"Adam: Param {i}, grad norm: {np.linalg.norm(grad):.8f}, "
            #       f"m_hat norm: {np.linalg.norm(m_hat):.8f}, sqrt(v_hat) norm: {np.linalg.norm(np.sqrt(v_hat)):.8f}, "
            #       f"update norm: {np.linalg.norm(update):.8f}")
            p['param'] -= update

    def zero_grad(self):
        for p in self.params:
            if p['grad'] is not None:
                p['grad'] = np.zeros_like(p['param'])
