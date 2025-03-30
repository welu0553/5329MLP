from collections import OrderedDict

class Module:
    """
    统一的接口和参数管理类
    所有后续实现的层（eg: Linear, ReLU, Dropout, BatchNorm etc.）都应继承此基类
    """
    def __init__(self):
        # 训练模式（默认）
        self.training = True
        # 子模块存储单元
        self._modules = {}

    def forward(self, x):
        """
        向前传播，必须重写

        :param x:
        :return:
        """
        raise NotImplementedError('方法缺失，子类必须重写forward方法！')

    def __call__(self, x):
        """
        重载，使类实例可以直接调用forward func
        :param x:
        :return:
        """
        return self.forward(x)

    def parameters(self):
        """
        recursion收集本模块及子类的所有可训练参数
        :return: list，每个元素是一个numpy array
        """
        params = []
        # W: weight
        if hasattr(self, 'W'):
            params.append(self.W)
        # b: bias
        if hasattr(self, 'b'):
            params.append(self.b)

        # 收集子类模块的参数
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_module(self, name, module):
        """
        将子模块添加到本模块中
        :param name:
        :param module:
        :return:
        """
        self._modules[name] = module

    def train(self):
        """
        设置本模块及所有子模块为train模式
        """
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        """
        设置本模块及所有子模块为评估模式
        """
        self.training = False
        for module in self._modules.values():
            module.eval()

class Sequential(Module):
    """
    主要目的:
        将多个模块（都继承自 Module 基类）
        按顺序组合在一起，从而简化整体模型的构建和前向传播流程
    """
    def __init__(self, *args):
        super().__init__()
        """
        Sequential 可以允许接收任何数量的模块。
        当只传入一个模块的时候，则直接使用它，
        若传入多个模块，则将它们添加到一个 OrderedDict 中，
        key 为字符串形式的 index
        """
        # 这里使用了 OrderedDict 而不是 list, 为的是方便后期调试或扩展
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            self.modules = args[0]
        else:
            self.modules = OrderedDict()
            for index, module in enumerate(args):
                self.modules[str(index)] = module

    def forward(self, x):
        # 依次调用各子模块的 forward 方法
        for name, module in self.modules.items():
            if hasattr(module, 'training'):
                module.training = self.training
            x = module(x)
        return x

    def backward(self, grad_output, lr=None):
        """
        反向传播，逆序调用各子模块的 backward 方法。
        注意：如果子模块的 backward 需要 lr 参数（例如 Linear 层），则传入 lr。
        """
        # 逆序遍历子模块
        for name, module in reversed(self.modules.items()):
            # 如果模块有 backward 方法，检查它是否需要 lr 参数
            if hasattr(module, 'backward'):
                # 这里做一个简单判断：若模块类型是 Linear，则传入 lr，否则只传入 grad_output
                if hasattr(module, 'W'):  # 假设 Linear 层有 W 属性
                    grad_output = module.backward(grad_output, lr)
                else:
                    grad_output = module.backward(grad_output)
        return grad_output


    def parameters(self):
        # 递归收集各子模块的可训练参数
        params = []
        for name, module in self.modules.items():
            params.extend(module.parameters())
        return params


    def train(self):
        self.training = True
        for module in self.modules.values():
            if hasattr(module, 'train'):
                module.train()

    def eval(self):
        self.training = False
        for module in self.modules.values():
            if hasattr(module, 'eval'):
                module.eval()