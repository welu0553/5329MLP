import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from data_loader import DataLoader
from models import MLP
from utils import load_model, assign_parameters, testing_model, training_model, save_model


# Generate a unique filename using a timestamp

def generate_model_filename(prefix="relu", extension="npz"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

# Create all combinations of hyperparameters and optimizer-specific parameters

def product(lists):
    result = [[]]
    for lst in lists:
        new_result = []
        for r in result:
            for item in lst:
                new_result.append(r + [item])
        result = new_result
    return result

def permutations_params(pool):
    pool = pool.copy()
    if all(k in pool for k in ['opt_para_SGD', 'opt_para_Adam', 'opt']):
        sgd_params = pool.pop('opt_para_SGD')
        adam_params = pool.pop('opt_para_Adam')
        base_keys = list(pool.keys())
        base_values = [pool[k] for k in base_keys]
        # generate all combinations
        base_combinations = product(base_values)
        result = []
        for comb in base_combinations:
            base_dict = dict(zip(base_keys, comb))
            if base_dict['opt'] == 'SGD':
                for p in sgd_params:
                    temp = base_dict.copy()
                    temp['opt_para'] = p
                    result.append(temp)
            elif base_dict['opt'] == 'Adam':
                for p in adam_params:
                    temp = base_dict.copy()
                    temp['opt_para'] = p
                    result.append(temp)
        return result
    else:
        keys = list(pool.keys())
        values = [pool[k] for k in keys]
        # generate all combinations, and transfer them to dictionary
        return [dict(zip(keys, comb)) for comb in product(values)]

# Train and save models using all combinations of hyperparameters

def train_all_models():
    loader = DataLoader(
        './Assignment1-Dataset/train_data.npy',
        './Assignment1-Dataset/train_label.npy',
        './Assignment1-Dataset/test_data.npy',
        './Assignment1-Dataset/test_label.npy',
        num_classes=10
    )

    hyperparams_pool = {
        'input_dim': [128],
        'hidden_dims': [[256, 128], [512, 256], [1024, 512, 256]],
        'output_dim': [10],
        'activation': ['relu'],
        'dropout_prob': [0.0, 0.1],
        'use_batchnorm': [True]
    }
    para_grad_pool = {
        'learning_rate': [0.005, 0.01],
        'batch_size': [512, 256],
        'num_epochs': [200],
        'opt': ['SGD', 'Adam'],
        'opt_para_SGD': [{'momentum': 0.9, 'weight_decay': 0.0001}, {'momentum': 0.92, 'weight_decay': 0.0001}],
        'opt_para_Adam': [{'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-6}, {'beta1': 0.92, 'beta2': 0.98, 'eps': 1e-6}],
        'shuffle': [True]
    }
    hyperparams_list = permutations_params(hyperparams_pool)
    para_grad_list = permutations_params(para_grad_pool)

    print(f"Total combinations: {len(hyperparams_list) * len(para_grad_list)}")

    for i, hyperparams in enumerate(hyperparams_list):
        for j, para_grad in enumerate(para_grad_list):
            model = MLP(**hyperparams)
            training_model(model, loader, para_grad)
            save_model(model, {'hyperparams': hyperparams, 'para_grad': para_grad},
                       f"./saves/{generate_model_filename()}")

# Evaluate all models and return the best performing one based on Macro-F1

def evaluate_all_models():
    loader = DataLoader(
        './Assignment1-Dataset/train_data.npy',
        './Assignment1-Dataset/train_label.npy',
        './Assignment1-Dataset/test_data.npy',
        './Assignment1-Dataset/test_label.npy',
        num_classes=10
    )

    model_dir = './saves'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.npz')]
    results = []

    for model_file in model_files:
        model_params, loaded = load_model(os.path.join(model_dir, model_file))
        model = MLP(**loaded['hyperparams'])
        assign_parameters(model, model_params)
        result = testing_model(model, loader)
        acc, f1 = result[0], result[4]
        results.append((model_file, acc, f1, loaded['hyperparams'], loaded['para_grad']))

    results.sort(key=lambda x: x[2], reverse=True)
    best = results[0]
    best_name, best_acc, best_f1, best_hyper, best_grad = best

    # Save evaluation results for all models
    with open("all_model_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Macro-F1', 'Hyperparams', 'TrainingParams'])
        for r in results:
            writer.writerow([r[0], r[1], r[2], str(r[3]), str(r[4])])

    print(f"\n Tested all {len(results)} models.")
    print(f"Best Model: {best_name}\nAccuracy: {best_acc:.4f}, Macro-F1: {best_f1:.4f}\nHyperparams: {best_hyper}\nTrain Params: {best_grad}")

    # Visualize best model performance
    plt.figure(figsize=(6, 4))
    plt.plot(['Accuracy', 'Macro-F1'], [best_acc, best_f1], marker='o', linewidth=2, color='royalblue')
    plt.title('Best Model Performance', fontsize=12)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('best_model_result.png')

    return best_hyper, best_grad

# Run ablation experiments on variants of the best model

def run_ablation_experiment(best_hyper, best_grad):
    loader = DataLoader(
        './Assignment1-Dataset/train_data.npy',
        './Assignment1-Dataset/train_label.npy',
        './Assignment1-Dataset/test_data.npy',
        './Assignment1-Dataset/test_label.npy',
        num_classes=10
    )

    variants = [
        ('Original', best_hyper),
        ('No Dropout', {**best_hyper, 'dropout_prob': 0.0}),
        ('No BatchNorm', {**best_hyper, 'use_batchnorm': False}),
        ('GELU Activation', {**best_hyper, 'activation': 'gelu'})
    ]

    ablation_results = []
    for name, hparams in variants:
        model = MLP(**hparams)
        training_model(model, loader, best_grad)
        result = testing_model(model, loader)
        acc, f1 = result[0], result[4]
        ablation_results.append((name, acc, f1, hparams))

    names = [r[0] for r in ablation_results]
    accs = [r[1] for r in ablation_results]
    f1s = [r[2] for r in ablation_results]
    params = [str(r[3]) for r in ablation_results]

    # Save and visualize ablation experiment results
    plt.figure(figsize=(10, 6))
    x = np.arange(len(names))
    plt.plot(x, accs, marker='o', label='Accuracy', linewidth=2)
    plt.plot(x, f1s, marker='s', label='Macro-F1', linewidth=2)
    plt.xticks(x, names, rotation=15, fontsize=10)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.title('Ablation Study on Best Model', fontsize=14)

    # Annotate each point with accuracy and f1
    for i, (acc, f1) in enumerate(zip(accs, f1s)):
        plt.text(i, acc + 0.02, f"Acc: {acc:.4f}", ha='center', fontsize=9, color='blue')
        plt.text(i, f1 + 0.02, f"F1: {f1:.4f}", ha='center', fontsize=9, color='darkorange')

    plt.tight_layout()
    plt.savefig('ablation_result.png')
    print("Ablation experiment results saved as ablation_result.png")

    # Print table-like output in terminal
    print("\n===== Ablation Results =====")
    for name, acc, f1, hparam in ablation_results:
        print(f"{name:<15} | Acc: {acc:.4f} | F1: {f1:.4f} | Hyperparams: {hparam}")

# Run full pipeline
if __name__ == '__main__':
    # Step 1: Train all models with different hyperparameter settings
    #train_all_models()
    # Step 2: Evaluate all trained models and get the best configuration
    best_hyper, best_grad = evaluate_all_models()
    # Step 3: Perform ablation study on the best model
    run_ablation_experiment(best_hyper, best_grad)

