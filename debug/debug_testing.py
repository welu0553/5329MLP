from data_loader import DataLoader
from models.mlp_model import MLP
from utils import load_model, testing_model, assign_parameters

# Load dataset
train_data_path = '../Assignment1-Dataset/train_data.npy'
train_label_path = '../Assignment1-Dataset/train_label.npy'
test_data_path = '../Assignment1-Dataset/test_data.npy'
test_label_path = '../Assignment1-Dataset/test_label.npy'
loader = DataLoader(train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10)

# Load trained model
model_params, loaded_params = load_model("../saves/model_test.npz")
hyperparams = loaded_params['hyperparams']
print("Loaded hyperparameters:", hyperparams)

# Rebuild model using saved hyperparameters and assign loaded weights
model = MLP(**hyperparams)
assign_parameters(model, model_params)

# ---------------------
# Evaluate model on test set
result = testing_model(model, loader)
# Returns: acc, precisions, recalls, f1s, macro_f1, confusion_matrix
[print(x, '\n') for x in result]
