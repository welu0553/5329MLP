import numpy as np
import time
from losses import Softmax, CrossEntropyLoss
from optimizers import SGD, Adam


def training_model(model, data_loader, para_grad):
    model.train()

    learning_rate = para_grad['learning_rate']
    batch_size = para_grad['batch_size']
    num_epochs = para_grad['num_epochs']
    opt = para_grad['opt']
    opt_para = para_grad['opt_para']
    data_shuffle = para_grad['shuffle']

    # Initialize loss and optimizer
    softmax = Softmax()
    criterion = CrossEntropyLoss()
    params = model.parameters()

    if opt == 'SGD':
        optimizer = SGD(params, lr=learning_rate,
                        momentum=opt_para['momentum'], weight_decay=opt_para['weight_decay'])
    elif opt == 'Adam':
        optimizer = Adam(params, lr=learning_rate,
                         beta1=opt_para['beta1'], beta2=opt_para['beta2'], eps=opt_para['eps'])
    else:
        raise TypeError('Wrong Input!!!')

    # Logging variables for debugging
    loss_history = []
    grad_norm_history = []
    param_norm_change_history = []  # Record the average parameter update changes for each epoch

    # ---------------------
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_grad_norms = []
        param_norm_changes = []
        start_time = time.time()
        num_batches = 0

        for X_batch, y_batch in data_loader.batch_generator(mode='train', batch_size=batch_size, shuffle=data_shuffle):
            num_batches += 1
            # Record the norm of all parameters before updating
            pre_param_norms = [np.linalg.norm(p['param']) for p in model.parameters()]

            # Forward Propagation
            logits = model(X_batch)
            probs = softmax.forward(logits)
            loss = criterion.forward(probs, y_batch)
            epoch_loss += loss

            # Backpropagation: Gradients are calculated and
            # stored in the persistent parameter dictionary of each layer
            grad_logits = criterion.backward(probs, y_batch)
            model.backward(grad_logits)  # Each layer only calculates and saves the gradient

            # Record the average norm of the current mini-batch gradient
            batch_norms = []
            for p in model.parameters():
                if p['grad'] is not None:
                    batch_norms.append(np.linalg.norm(p['grad']))
            if batch_norms:
                batch_grad_norms.append(np.mean(batch_norms))

            # Update Parameters
            optimizer.step()

            # Record the L2 norm of all parameters after update
            post_param_norms = [np.linalg.norm(p['param']) for p in model.parameters()]
            # Calculate the updated change (absolute value) of each parameter and then average it
            batch_change = np.mean([abs(post - pre) for pre, post in zip(pre_param_norms, post_param_norms)])
            param_norm_changes.append(batch_change)
            if batch_change < 1e-12:
                print(f"DEBUG: Epoch {epoch + 1}, Batch {num_batches} param change is nearly 0")

            # Zero gradient
            optimizer.zero_grad()

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        avg_grad_norm = np.mean(batch_grad_norms) if batch_grad_norms else 0
        grad_norm_history.append(avg_grad_norm)
        avg_param_change = np.mean(param_norm_changes) if param_norm_changes else 0
        param_norm_change_history.append(avg_param_change)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.6f}, "
              f"Avg Param Change: {avg_param_change:.8f}, Time: {time.time() - start_time:.2f}s")

    return loss_history, grad_norm_history, param_norm_change_history
