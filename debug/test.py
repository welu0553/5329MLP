import time
from data_loader import DataLoader
from models.mlp_model import MLP
from losses import Softmax, CrossEntropyLoss
from optimizers import SGD  # or use Adam

# Hyperparameter settings
learning_rate = 0.05
batch_size = 32
num_epochs = 10

# Data paths (adjust paths as needed)
train_data_path = '../Assignment1-Dataset/train_data.npy'
train_label_path = '../Assignment1-Dataset/train_label.npy'
test_data_path = '../Assignment1-Dataset/test_data.npy'
test_label_path = '../Assignment1-Dataset/test_label.npy'

# Load data
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10)
print(loader)

# Build the model
# Assume input feature size is obtained from the shape of training data
input_dim = loader.get_train_data().shape[1]
hidden_dims = [128, 64]  # Example: two hidden layers
output_dim = 10  # 10-class classification problem

model = MLP(input_dim, hidden_dims, output_dim, activation='relu', dropout_prob=0.5, use_batchnorm=True)

# Initialize softmax and loss function
softmax = Softmax()
criterion = CrossEntropyLoss()

# Collect trainable model parameters (each returned as {'param': ..., 'grad': ...})
params = model.parameters()

# Initialize optimizer (using SGD; can be switched to Adam)
optimizer = SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    start_time = time.time()
    # Use mini-batch generator, mode set to 'train', shuffle enabled
    for X_batch, y_batch in loader.batch_generator(mode='train', batch_size=batch_size, shuffle=True):
        # Forward pass: compute logits
        logits = model(X_batch)
        # Compute softmax probabilities
        probs = softmax.forward(logits)
        # Compute cross-entropy loss (y_batch is already one-hot encoded)
        loss = criterion.forward(probs, y_batch)
        epoch_loss += loss

        # Backward pass: compute gradient of cross-entropy, then backpropagate through model
        grad_logits = criterion.backward(probs, y_batch)
        # Backpropagation: layers store gradients internally
        model.backward(grad_logits)

        # Update parameters using optimizer
        optimizer.step()
        # Clear gradients for next iteration
        optimizer.zero_grad()

    avg_loss = epoch_loss / (loader.get_train_data().shape[0] / batch_size)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

