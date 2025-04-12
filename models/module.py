from collections import OrderedDict

class Module:
    """
    Base class for all neural network layers.

    Attributes:
        training (bool): Whether the module is in training mode. Default is True.
        _modules (dict): A dictionary to hold submodules added via add_module().
    """

    def __init__(self):
        self.training = True
        self._modules = {}

    def forward(self, x):
        """
        Forward pass. Must be overridden by subclasses.

        Args:
            x (np.ndarray): Input tensor.
        Returns:
            np.ndarray: Output tensor.
        """
        raise NotImplementedError('Missing method: subclasses must override forward().')

    def __call__(self, x):
        """
        Make the instance callable. This calls the forward() method.

        Args:
            x (np.ndarray): Input tensor.
        Returns:
            np.ndarray: Output tensor from forward pass.
        """
        return self.forward(x)

    def parameters(self):
        """
        Recursively collect all trainable parameters from this module and submodules.

        Returns:
            list: A list of numpy arrays representing parameters (e.g. weights and biases).
        """
        params = []
        if hasattr(self, 'W'):
            params.append(self.W)
        if hasattr(self, 'b'):
            params.append(self.b)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_module(self, name, module):
        """
        Add a submodule to this module.

        Args:
            name (str): Name of the submodule.
            module (Module): The submodule instance to add.
        """
        self._modules[name] = module

    def train(self):
        """
        Set this module and all its submodules to training mode.
        """
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        """
        Set this module and all its submodules to evaluation mode.
        """
        self.training = False
        for module in self._modules.values():
            module.eval()


class Sequential(Module):
    """
    A container to wrap multiple modules into a sequence.

    Purpose:
        Automatically chain modules together in the order they are added,
        simplifying model building and forward/backward propagation.

    Attributes:
        modules (OrderedDict): A dictionary mapping names to child modules, in order.
    """

    def __init__(self, *args):
        super().__init__()
        # Accept a single OrderedDict or multiple modules as args
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            self.modules = args[0]
        else:
            self.modules = OrderedDict()
            for index, module in enumerate(args):
                self.modules[str(index)] = module

    def forward(self, x):
        """
        Forward pass through all submodules in sequence.

        Args:
            x (np.ndarray): Input tensor.
        Returns:
            np.ndarray: Output tensor after all layers.
        """
        for name, module in self.modules.items():
            if hasattr(module, 'training'):
                module.training = self.training
            x = module(x)
        return x

    def backward(self, grad_output, lr=None):
        """
        Backward pass through all submodules in reverse order.

        Args:
            grad_output (np.ndarray): Upstream gradient.
            lr (float or None): Learning rate if needed by certain layers (e.g., Linear).
        Returns:
            np.ndarray: Gradient w.r.t. input.
        """
        for name, module in reversed(self.modules.items()):
            if hasattr(module, 'backward'):
                # If it's a Linear layer (detected via 'W'), pass learning rate
                if hasattr(module, 'W'):
                    grad_output = module.backward(grad_output, lr)
                else:
                    grad_output = module.backward(grad_output)
        return grad_output

    def parameters(self):
        """
        Recursively collect parameters from all submodules.

        Returns:
            list: All trainable parameters from the sequence.
        """
        params = []
        for name, module in self.modules.items():
            params.extend(module.parameters())
        return params

    def train(self):
        """
        Set all modules to training mode.
        """
        self.training = True
        for module in self.modules.values():
            if hasattr(module, 'train'):
                module.train()

    def eval(self):
        """
        Set all modules to evaluation mode.
        """
        self.training = False
        for module in self.modules.values():
            if hasattr(module, 'eval'):
                module.eval()
