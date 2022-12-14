"""
This script is used to evaluate Aster with chosen dataset and target model
"""
import pickle
import argparse
import numpy as np
import torch
from data_preprocessing import data_reader
from computation_utils import Target_Model_pred_fn
from computation_utils import fn_R_given_Selected
from computation_utils import fn_Sample_Generator
from computation_utils import fn_Jacobian_Calculation

data = []


def run(dataset, model, epsilon, training_samples, seed, label=-1):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attack', type=int, default=training_samples)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--neighbors', type=int, default=40)
    parser.add_argument('--data_generate', type=bool, default=False)
    attack_args = parser.parse_args()

    result = []
    np.random.seed(seed=attack_args.seed)
    torch.manual_seed(attack_args.seed)
    filename = dataset + "_" + model + ".pkl"

    # load data
    orig_dataset, oh_dataset, OH_Encoder = data_reader(dataset, training_samples)
    class_label_for_count = np.unique(np.hstack([orig_dataset["Y_train"], orig_dataset["Y_test"]]))

    n_class = len(class_label_for_count)
    n_features = orig_dataset['X_train'].shape[1]
    print(n_features)
    Target_Model = None
    # load pretrained target model
    with open('./target_models/' + filename, 'rb') as f:
        Target_Model = pickle.load(f)

    y_attack = np.hstack(([np.ones(int(attack_args.n_attack / 2)), np.zeros(int(attack_args.n_attack / 2))]))
    x_attack = np.zeros((int(attack_args.n_attack), n_features))
    Jacobian_matrix = np.zeros([attack_args.n_attack, n_class, n_features])

    if attack_args.data_generate:
        output_x = np.zeros((attack_args.n_attack, n_features))
        output_y = y_attack
        classes = np.zeros((attack_args.n_attack, 1))

    for ii in range(attack_args.n_attack):
        R_x, R_y = fn_R_given_Selected(orig_dataset, IN_or_OUT=y_attack[ii])
        R_x_OH = OH_Encoder.transform(R_x.reshape(1, -1))
        x_attack[ii] = R_x
        local_samples = fn_Sample_Generator(R_x, dataset, epsilon)
        oh_local_samples = OH_Encoder.transform(local_samples)
        local_proba = Target_Model_pred_fn(Target_Model, oh_local_samples)
        R_local_proba = Target_Model_pred_fn(Target_Model, R_x_OH)
        Jacobian_matrix[ii] = fn_Jacobian_Calculation(R_local_proba[0], local_proba, n_features, n_class)

    if label == -1:
        for matrix in Jacobian_matrix:
            data.append(matrix.flatten())
    elif label == 0:
        for matrix in Jacobian_matrix:
            data.append(np.append(matrix.flatten(), 0))
    else:
        for matrix in Jacobian_matrix:
            data.append(np.append(matrix.flatten(), 1))

    return data