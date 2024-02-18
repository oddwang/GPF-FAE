import numpy as np

from synthetic_dataset import generate_synthetic_data
from GPF_FAE_metric import data_encode, construct_model,GPF_FAE_metric

from torch_two_sample import *
from scipy.spatial import distance
import torch

def MMD_each_feature(male_explains, female_explains):
    p_vals = []
    for i in range(len(male_explains[0])):
        male_explain = male_explains[:, i]
        female_explain = female_explains[:, i]

        mmd_test = MMDStatistic(len(male_explain), len(female_explain))

        if len(male_explain.shape) == 1:
            male_explain = male_explain.reshape((len(male_explain), 1))
            female_explain = female_explain.reshape((len(female_explain), 1))
            all_dist = distance.cdist(male_explain, male_explain, 'euclidean')
        else:
            all_dist = distance.cdist(male_explain, female_explain, 'euclidean')
        median_dist = np.median(all_dist)

        # Calculate MMD.
        t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(male_explain)),
                                 torch.autograd.Variable(torch.tensor(female_explain)),
                                 alphas=[1 / median_dist], ret_matrix=True)
        p_val = mmd_test.pval(matrix)
        p_vals.append(p_val)
    p_vals = np.array(p_vals)
    return p_vals

def detect_unfair_features(D1_select_explain_result, D2_select_explain_result, alpha=0.05):
    each_feature_MMD_value = MMD_each_feature(D1_select_explain_result, D2_select_explain_result)
    unfair_features = np.where(each_feature_MMD_value < alpha)

    return unfair_features[0]


if __name__ == '__main__':
    # Using the synthetic dataset as an example, you can replace it with the dataset you wish to use
    X, sensitive_feature_idx, y = generate_synthetic_data(-0.2, 1.5, 0.5, 0.5, 0.5)

    # Data pre-process
    X_train, X_test, y_train, y_test = data_encode(X, sensitive_feature_idx, y)

    sensitive_feature_train = X_train[:, sensitive_feature_idx]
    sensitive_feature_test = X_test[:, sensitive_feature_idx]

    # Constructing the ML model to be explained
    model = construct_model(X_train, y_train)

    # Evaluating the procedural fairness of the model with the GPF_FAE metric
    D1_select, D2_select, D1_select_explain_result, D2_select_explain_result, GPF_FAE_result = \
        GPF_FAE_metric(X_train, X_test, model, sensitive_feature_test, n=100)

    unfair_features = detect_unfair_features(D1_select_explain_result, D2_select_explain_result)

    print('unfair feature index: ', unfair_features)
