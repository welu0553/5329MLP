import numpy as np


class Optimizer:
    '''Base class for all optimizers'''

    def __init__(self, params, lr=0.01):
        """
        :param params: List of parameter dictionaries. Each dict must contain:
            - 'param': the parameter array (np.ndarray)
            - 'grad': the gradient array (np.ndarray)
        :param lr: Learning rate (float)
        """
        self.params = params
        self.lr = lr

    def step(self):
        '''Perform a single optimization step (to be implemented in subclass)'''
        raise NotImplementedError

    def zero_grad(self):
        '''Reset all gradients to zero'''
        for p in self.params:
            p['grad'] = np.zeros_like(p['grad'])


class SGD(Optimizer):
    '''Stochastic Gradient Descent optimizer with optional momentum and weight decay'''

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        :param momentum: Momentum factor (default 0.0)
        :param weight_decay: L2 regularization factor (default 0.0)
        """
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize momentum buffer (velocity) for each parameter
        self.velocities = [np.zeros_like(p['param'], dtype=np.float64) for p in self.params]

    def step(self):
        '''Apply one step of SGD update with momentum and weight decay'''
        for i, p in enumerate(self.params):
            if p['grad'] is None:
                continue
            # Ensure parameter is float64 for numerical precision
            p['param'] = np.asarray(p['param'], dtype=np.float64)
            # Apply L2 regularization if weight decay is non-zero
            grad = p['grad'] + self.weight_decay * p['param']
            # Update momentum buffer
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            self.velocities[i] = np.asarray(self.velocities[i], dtype=np.float64)
            # In-place parameter update
            p['param'][:] += self.velocities[i]


class Adam(Optimizer):
    '''Adaptive Moment Estimation (Adam) optimizer'''

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-6):
        """
        :param beta1: Exponential decay rate for the first moment estimate
        :param beta2: Exponential decay rate for the second moment estimate
        :param eps: A small constant to prevent division by zero
        """
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Time step counter
        # Initialize first and second moment estimates
        self.m = [np.zeros_like(p['param'], dtype=np.float64) for p in self.params]
        self.v = [np.zeros_like(p['param'], dtype=np.float64) for p in self.params]

    def step(self):
        '''Apply one step of Adam optimization'''
        self.t += 1
        for i, p in enumerate(self.params):
            if p['grad'] is None:
                continue
            grad = p['grad']
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Compute parameter update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            # Apply update
            p['param'] -= update

    def zero_grad(self):
        '''Reset gradients to zero for all parameters'''
        for p in self.params:
            if p['grad'] is not None:
                p['grad'] = np.zeros_like(p['param'])
