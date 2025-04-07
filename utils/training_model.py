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

    # 初始化损失和优化器
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

    # 用于调试的记录变量
    loss_history = []
    grad_norm_history = []
    param_norm_change_history = []  # 记录每个 epoch 平均参数更新变化

    # ---------------------
    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_grad_norms = []
        param_norm_changes = []
        start_time = time.time()
        num_batches = 0

        for X_batch, y_batch in data_loader.batch_generator(mode='train', batch_size=batch_size, shuffle=data_shuffle):
            num_batches += 1
            # 记录更新前所有参数的 L2 范数
            pre_param_norms = [np.linalg.norm(p['param']) for p in model.parameters()]

            # 前向传播
            logits = model(X_batch)
            probs = softmax.forward(logits)
            loss = criterion.forward(probs, y_batch)
            epoch_loss += loss

            # 反向传播：计算梯度并存储到各层的持久参数字典中
            grad_logits = criterion.backward(probs, y_batch)
            model.backward(grad_logits)  # 各层只计算并保存梯度

            # 记录当前 mini-batch 梯度的平均范数
            batch_norms = []
            for p in model.parameters():
                if p['grad'] is not None:
                    batch_norms.append(np.linalg.norm(p['grad']))
            if batch_norms:
                batch_grad_norms.append(np.mean(batch_norms))

            # 更新参数
            optimizer.step()

            # 记录更新后所有参数的 L2 范数
            post_param_norms = [np.linalg.norm(p['param']) for p in model.parameters()]
            # 计算每个参数的更新变化（绝对值），然后求平均
            batch_change = np.mean([abs(post - pre) for pre, post in zip(pre_param_norms, post_param_norms)])
            param_norm_changes.append(batch_change)
            if batch_change < 1e-12:
                print(f"DEBUG: Epoch {epoch + 1}, Batch {num_batches} param change is nearly 0")

            # 清零梯度
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
