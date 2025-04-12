import numpy as np
from data_loader import DataLoader
from models.mlp_model import MLP
from losses import Softmax, CrossEntropyLoss
# Set data paths and configuration (modify as needed)
train_data_path = '../Assignment1-Dataset/train_data.npy'
train_label_path = '../Assignment1-Dataset/train_label.npy'
test_data_path = '../Assignment1-Dataset/test_data.npy'
test_label_path = '../Assignment1-Dataset/test_label.npy'

num_classes = 10

# Initialize data loader (will normalize data and convert labels to one-hot encoding)
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=num_classes)
print(loader)  # Print dataset statistics

# Fetch a mini-batch from the batch generator
batch_gen = loader.batch_generator(mode='train', batch_size=32, shuffle=True)
X_batch, y_batch = next(batch_gen)
print("Mini-batch data shape:", X_batch.shape)
print("Mini-batch label shape:", y_batch.shape)

# Build the MLP model
input_dim = loader.get_train_data().shape[1]
hidden_dims = [128, 64]
output_dim = num_classes
model = MLP(input_dim, hidden_dims, output_dim, activation='relu', dropout_prob=0.5, use_batchnorm=True)

# Initialize Softmax and CrossEntropy loss modules
softmax = Softmax()
criterion = CrossEntropyLoss()

# --- Forward pass ---
logits = model(X_batch)
probs = softmax.forward(logits)
loss = criterion.forward(probs, y_batch)
print("Initial Loss: {:.4f}".format(loss))

# --- Backward pass ---
grad_logits = criterion.backward(probs, y_batch)
# Note: lr is not passed here, so layers only compute and store gradients
model.backward(grad_logits)

# --- Gradient check for each layer ---
params = model.parameters()  # Each item is a dict: {'param': ..., 'grad': ...}
print("\nGradient check:")
for i, p in enumerate(params):
    grad = p.get('grad', None)
    if grad is None:
        print(f"Parameter {i}: shape {p['param'].shape}, grad is None")
    else:
        norm = np.linalg.norm(grad)
        print(f"Parameter {i}: shape {p['param'].shape}, grad norm {norm:.6f}")

# If gradients are extremely close to zero or very large,
# you may need to debug the backward implementations of specific layers,
# or adjust hyperparameters like learning rate or initialization method.
