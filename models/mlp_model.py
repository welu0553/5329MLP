from collections import OrderedDict
from models.module import Module
from models.layers import Linear, ReLU, Dropout, BatchNorm, GELU
from models.module import Sequential

class MLP(Module):
    """
    A multi-layer perceptron (MLP) model with optional BatchNorm, Dropout, and selectable activation.

    Attributes:
        model (Sequential): A sequential container of layers including Linear, BatchNorm, Activation, and Dropout layers.
    """

    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation='relu', dropout_prob=0.0, use_batchnorm=False):
        """
        Initialize the MLP architecture with specified configuration.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dims (list of int): List of hidden layer sizes, e.g. [128, 64].
            output_dim (int): Output feature dimension (number of classes).
            activation (str): Activation function name ('relu' or 'gelu').
            dropout_prob (float): Dropout probability. If 0, dropout is disabled.
            use_batchnorm (bool): Whether to use Batch Normalization after each hidden layer.
        """
        super().__init__()
        layers = OrderedDict()
        prev_dim = input_dim

        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Add Linear layer
            layers[f'linear{i}'] = Linear(prev_dim, hidden_dim)
            # Optional: add Batch Normalization after Linear layer
            if use_batchnorm:
                layers[f'batchnorm{i}'] = BatchNorm(hidden_dim)
            # Add activation layer
            if activation.lower() == 'relu':
                layers[f'relu{i}'] = ReLU()
            elif activation.lower() == 'gelu':
                layers[f'gelu{i}'] = GELU()
            # Optional: add Dropout after activation
            if dropout_prob > 0:
                layers[f'dropout{i}'] = Dropout(dropout_prob)
            prev_dim = hidden_dim

        # Add output layer
        layers['linear_out'] = Linear(prev_dim, output_dim)

        # Pack all layers into a Sequential container
        self.model = Sequential(layers)

    def forward(self, x):
        """
        Forward pass through the entire MLP.
        Args:
            x (np.ndarray): Input tensor.
        Returns:
            np.ndarray: Output logits.
        """
        return self.model(x)

    def backward(self, grad_output, lr=None):
        """
        Backward pass through the entire MLP.
        Args:
            grad_output (np.ndarray): Gradient from loss layer.
            lr (float or None): If provided, apply immediate weight update.
        """
        self.model.backward(grad_output, lr)

    def parameters(self):
        """
        Collect all trainable parameters and gradients from each layer.
        Returns:
            list of dict: Each dict contains 'param' and 'grad'.
        """
        return self.model.parameters()

    def train(self):
        """
        Set all modules to training mode (e.g., enabling dropout and batchnorm statistics).
        """
        self.model.train()

    def eval(self):
        """
        Set all modules to evaluation mode (e.g., disable dropout and use running stats in batchnorm).
        """
        self.model.eval()
