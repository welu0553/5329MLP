# 5329MLP Project Documentation
### 1. Project Description

This project implements a modular MLP (Multi-Layer Perceptron) neural network training platform based on Python and Numpy, supporting the entire process from data loading, model building, batch training, automatic evaluation, model selection, to ablation experiments. The core functions of this project are divided into the following parts:

Data import and processing Supports data files in .npyformat as input; supports loading of training and test sets; automatically handles basic preprocessing operations such as data normalisation and label conversion; uses a custom DataLoader class to encapsulate data batch processing logic, and supports the setting of parameters such as batch_size and shuffle.
Model structure construction The model is implemented based on a custom Module base class and Sequential container; supports user-defined MLP network layers, the dimension of each layer (hidden_dims), the type of activation function, whether to use BatchNorm, whether to use Dropout, etc.; supports the following network components: fully connected layer (Linear) activation function: ReLU, GELU (implemented using an approximation function) Dropout layer Batch Normalization layer
Model training module automatically traverses all combinations of hyperparameters for batch training; supports SGD and Adam optimizers, and also supports parameter adjustment such as momentum, weight decay, beta1, beta2, eps, etc.; each model is automatically saved in .npzformat after training is complete; supports unified configuration of training parameters such as epochs, learning rate, batch size, optimizer, etc.
Model testing and evaluation automatically reads the saved model file and uniformly restores and reconstructs the model; The evaluation function in evaluation.pyis used to calculate the following metrics: Accuracy macro-F1 The evaluation results of all models are saved as a all_model_results.csvfile, which contains the model file name, accuracy, F1 score, hyperparameter information, etc.
Model selection and result output Automatically select the model with the highest macro-F1 score among all models as the ‘best model’; The accuracy, F1 score and corresponding hyperparameter configuration of the best model are output; the visualization is saved as an image best_model_result.png.
Ablation experiment Based on the structure of the ‘best model’, the following three structural ablation experiments are performed: Remove Dropout (set to 0) Turn off BatchNorm Replace the activation function from ReLU to GELU Each structural variant is reinitialised and trained to ensure the fairness of the experiment; After testing all variants, output is automatically generated to show the changes in accuracy and Macro-F1; The specific hyperparameter configuration of each ablation experiment is also output for easy tracking and reproduction.

### 2. Project Structure
```plaintext
project_root/
├── main.py               # Program entry. You can train or test the model by setting hyperparameters
├── data_loader.py        # Data loading module. Outputs the incoming data as an iterable object, and optionally uses mini-batch and onehot encoding
├── losses.py             # Loss function. Contains Softmax and CrossEntropy
├── optimizers.py         # Optimizer. Contains SGD, Adam
├── models/               # Model module
│   ├── __init__.py       # Module initialization
│   ├── module.py         # Basic model module. Contains the parent class Module for all layers and the Sequential class
│   ├── layers.py         # Implementation of layers. Contains Linear, ReLU, GELU, Dropout, BatchNorm
│   └── mlp_model.py      # MLP module. Used to implement various functionalities of MLP
├── utils/                # Utility functions
│   ├── __init__.py       # Module initialization
│   ├── evaluation.py     # Evaluation indicators
│   ├── functions.py      # Function file. Contains various helper functions to enhance code readability
│   ├── training_model.py # Model training script. Encapsulates the entire model training process for convenient use in the main function
│   └── testing_model.py  # Model testing script. Encapsulates the entire model testing process for convenient use in the main function
└── README.md             # Project documentation

```