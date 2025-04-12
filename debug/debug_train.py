import matplotlib.pyplot as plt
from utils.training_model import training_model
from data_loader import DataLoader
from models.mlp_model import MLP
from utils import save_model

# Data paths (modify according to actual directory structure)
train_data_path = './Assignment1-Dataset/train_data.npy'
train_label_path = './Assignment1-Dataset/train_label.npy'
test_data_path = './Assignment1-Dataset/test_data.npy'
test_label_path = './Assignment1-Dataset/test_label.npy'

# ---------------------
# Load data
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10)
print(loader)

# ---------------------
# Build the model
hyperparams = {
    'input_dim': loader.get_train_data().shape[1],
    'hidden_dims': [1024, 512, 256],
    'output_dim': 10,
    'activation': 'relu',
    'dropout_prob': 0.01,
    'use_batchnorm': True
}
model = MLP(**hyperparams)
print("Model built.")

# Training configuration
para_grad = {
    'learning_rate': 0.01,
    'batch_size': 512,
    'num_epochs': 5,
    'opt': 'SGD',  # You can switch to 'Adam'
    'opt_para': {'momentum': 0.9, 'weight_decay': 0.0001},
    # 'opt_para': {'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-6},
    'shuffle': True
}

# Train the model and collect training metrics
histories = training_model(model, loader, para_grad)
loss_history, grad_norm_history, param_norm_change_history = histories

# Visualize loss, gradient norm, and parameter change over epochs
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_history, marker='o')
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(grad_norm_history, marker='o', color='orange')
plt.title("Avg Gradient Norm over epochs")
plt.xlabel("Epoch")
plt.ylabel("Avg Grad Norm")

plt.subplot(1, 3, 3)
plt.plot(param_norm_change_history, marker='o', color='green')
plt.title("Avg Parameter Norm Change over epochs")
plt.xlabel("Epoch")
plt.ylabel("Avg Param Change")

plt.tight_layout()
plt.show()

# Save model and training configuration
save_paras = {
    'hyperparams': hyperparams,
    'para_grad': para_grad
}

# Save model and training configuration to file
save_model(model, save_paras, "saves/model_test.npz")
