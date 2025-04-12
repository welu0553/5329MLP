import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from models.mlp_model import MLP
from losses import Softmax, CrossEntropyLoss
from optimizers import SGD


def check_project():
    # ---------------------
    # Data paths (modify based on your actual dataset location)
    train_data_path = '../Assignment1-Dataset/train_data.npy'
    train_label_path = '../Assignment1-Dataset/train_label.npy'
    test_data_path = '../Assignment1-Dataset/test_data.npy'
    test_label_path = '../Assignment1-Dataset/test_label.npy'
    num_classes = 10

    # ---------------------
    # Load dataset
    print("Loading data ...")
    loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=num_classes)
    print(loader)

    # Retrieve one mini-batch from the generator
    batch_gen = loader.batch_generator(mode='train', batch_size=32, shuffle=True)
    X_batch, y_batch = next(batch_gen)
    print(f"Mini-batch shapes: X: {X_batch.shape}, y: {y_batch.shape}")

    # ---------------------
    # Build the model
    input_dim = loader.get_train_data().shape[1]
    hidden_dims = [128, 64]
    output_dim = num_classes
    model = MLP(input_dim, hidden_dims, output_dim, activation='relu', dropout_prob=0.5, use_batchnorm=True)
    print("Model built.")

    # ---------------------
    # Initialize loss function and softmax module
    softmax = Softmax()
    criterion = CrossEntropyLoss()

    # ---------------------
    # Forward pass
    logits = model(X_batch)
    probs = softmax.forward(logits)
    loss = criterion.forward(probs, y_batch)
    print(f"Initial Loss: {loss:.4f}")

    # ---------------------
    # Backward pass: compute gradients
    grad_logits = criterion.backward(probs, y_batch)
    model.backward(grad_logits)  # Each layer only computes and stores gradients, no parameter update

    # Check gradient norms of each parameter
    params = model.parameters()  # Each parameter is a dict: {'param': ..., 'grad': ...}
    print("\nGradient check:")
    for i, p in enumerate(params):
        grad = p.get('grad', None)
        if grad is None:
            print(f"Parameter {i}: shape {p['param'].shape}, grad is None")
        else:
            norm = np.linalg.norm(grad)
            print(f"Parameter {i}: shape {p['param'].shape}, grad norm: {norm:.6f}")

    # ---------------------
    # Record parameter norms before the update
    pre_param_norms = [np.linalg.norm(p['param']) for p in params]

    # Create optimizer, e.g., SGD
    optimizer = SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)

    # Perform one optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Record parameter norms after the update
    post_param_norms = [np.linalg.norm(p['param']) for p in params]
    print("\nParameter norm changes after one optimizer step:")
    for i, (pre, post) in enumerate(zip(pre_param_norms, post_param_norms)):
        print(f"Parameter {i}: pre-norm = {pre:.6f}, post-norm = {post:.6f}, change = {post - pre:.6f}")

    # ---------------------
    # Optional: plot gradient histograms of each parameter
    plt.figure(figsize=(12, 4))
    for i, p in enumerate(params):
        if p['grad'] is not None:
            plt.subplot(1, len(params), i+1)
            plt.hist(p['grad'].flatten(), bins=20)
            plt.title(f'Param {i} grad hist')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    check_project()
