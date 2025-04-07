import numpy as np


def save_model(model, hyperparams, filename):
    """
    保存模型参数和超参数到一个 .npz 文件中。

    参数：
      model: 模型实例，要求具有 parameters() 方法，该方法返回一个持久的字典列表，
             每个字典包含键 'param' 对应模型的参数（例如一个 numpy 数组）。
      hyperparams: 用于构造模型的超参数字典，例如 {'input_dim': ..., 'hidden_dims': [...], ...}
      filename: 保存文件的名称，例如 "model.npz"

    保存后文件中会包含键 "param_0", "param_1", ... 对应各层参数，以及键 "hyperparams" 保存超参数字典。
    """
    params = model.parameters()  # 例如：[{'param': array, 'grad': ...}, ...]
    save_dict = {}
    for i, p in enumerate(params):
        save_dict[f"param_{i}"] = p['param']
    save_dict["hyperparams"] = hyperparams
    np.savez(filename, **save_dict)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    加载保存的模型文件，返回模型参数列表和超参数字典。

    返回：
      loaded_params: 一个列表，每个元素为一个 numpy 数组（对应保存时的 "param_i"）
      hyperparams: 超参数字典
    """
    data = np.load(filename, allow_pickle=True)
    hyperparams = data["hyperparams"].item()  # 恢复为字典
    loaded_params = []
    i = 0
    while f"param_{i}" in data:
        loaded_params.append(data[f"param_{i}"])
        i += 1
    print(f"Model loaded from {filename}, found {i} parameters.")
    return loaded_params, hyperparams

def assign_parameters(model, loaded_params):
    """
    将加载的参数赋值给模型的各层参数。
    假设 model.parameters() 返回的是持久的参数字典列表，
    每个字典包含 'param' 键。这里用原地赋值确保更新原始数组。
    """
    current_params = model.parameters()
    for cp, lp in zip(current_params, loaded_params):
        cp['param'][:] = lp