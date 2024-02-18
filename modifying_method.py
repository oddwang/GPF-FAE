import tensorflow as tf
tf.enable_eager_execution()
config = tf.ConfigProto()
sess = tf.Session(config=config)

import numpy as np

from synthetic_dataset import generate_synthetic_data
from GPF_FAE_metric import data_encode, construct_model,GPF_FAE_metric
from detect_unfair_feature import detect_unfair_features
from explain_perturb import *

def modifying_method(X_train, X_test, y_train, y_test, model, D1_select_explain_result, D2_select_explain_result):
    # Detect unfair features
    unfair_features = detect_unfair_features(D1_select_explain_result, D2_select_explain_result)

    inputs = tf.convert_to_tensor(X_train, dtype=tf.float32)
    outputs = tf.convert_to_tensor(y_train, dtype=tf.float32)

    acc, e_loss, p_loss = wrap_adv_train(X_test, inputs, y_test, outputs, model, unfair_features,
                                         n_epochs=50, lr=10e-7, alpha=15, normalise=True)
    return model


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

    print("GPF_FAE Metric Before Modifying: ", GPF_FAE_result)

    # Modifying the ML model by reducing the impact of unfair features
    modify_model = modifying_method(X_train, X_test, y_train, y_test, model, D1_select_explain_result, D2_select_explain_result)

    # Evaluating the procedural fairness of the model with the GPF_FAE metric after reducing the impact of unfair features
    D1_select_modify, D2_select_modify, D1_select_explain_result_modify, D2_select_explain_result_modify, GPF_FAE_result_modify = \
        GPF_FAE_metric(X_train, X_test, modify_model, sensitive_feature_test, n=100)
    print("GPF_FAE Metric After Modifying: ", GPF_FAE_result_modify)

