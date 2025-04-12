import numpy as np


def save_model(model, hyperparams, filename):
    """
Save model parameters and hyperparameters to a .npz file.

Parameters:
    model: model instance, required to have a parameters() method,
        which returns a persistent list of dictionaries,
        each dictionary contains a key 'param'
        corresponding to the model's parameters (e.g. a numpy array).
    hyperparams: a hyperparameter dictionary used to
        construct the model, e.g. {'input_dim': ..., 'hidden_dims': [...], ...}
    filename: the name of the saved file, e.g. "model.npz"

After saving, the file will contain keys "param_0", "param_1", ... corresponding
to the parameters of each layer, and key "hyperparams" to save the hyperparameter dictionary.
    """
    params = model.parameters()  # eg：[{'param': array, 'grad': ...}, ...]
    save_dict = {}
    for i, p in enumerate(params):
        save_dict[f"param_{i}"] = p['param']
    save_dict["hyperparams"] = hyperparams
    np.savez(filename, **save_dict)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Load the saved model file and return the model parameter list and hyperparameter dictionary.

    Returns:
        loaded_params: a list, each element is a numpy array (corresponding to "param_i" when saving)
        hyperparams: hyperparameter dictionary
    """
    data = np.load(filename, allow_pickle=True)
    hyperparams = data["hyperparams"].item()  # Restore to Dictionary
    loaded_params = []
    i = 0
    while f"param_{i}" in data:
        loaded_params.append(data[f"param_{i}"])
        i += 1
    print(f"Model loaded from {filename}, found {i} parameters.")
    return loaded_params, hyperparams

def assign_parameters(model, loaded_params):
    """
    Assign the loaded parameters to the model's layer parameters.
    Assume that model.parameters() returns a persistent list of parameter dictionaries,
    each dictionary contains a 'param' key. Here, in-place assignment is
    used to ensure that the original array is updated.
    """
    current_params = model.parameters()
    for cp, lp in zip(current_params, loaded_params):
        cp['param'][:] = lp