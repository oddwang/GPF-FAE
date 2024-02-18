import numpy as np

from synthetic_dataset import generate_synthetic_data
from GPF_FAE_metric import data_encode, construct_model,GPF_FAE_metric
from detect_unfair_feature import detect_unfair_features

def retraining_method(X_train, X_test, y_train, D1_select_explain_result, D2_select_explain_result):
    # Detect unfair features
    unfair_features = detect_unfair_features(D1_select_explain_result, D2_select_explain_result)

    # Eliminate unfair features and reconstructing the ML model
    X_train_eliminate = np.delete(X_train, unfair_features, axis=1)
    X_test_eliminate = np.delete(X_test, unfair_features, axis=1)
    eliminate_model = construct_model(X_train_eliminate, y_train)

    return X_train_eliminate, X_test_eliminate, eliminate_model


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

    print("GPF_FAE Metric Before Retraining: ", GPF_FAE_result)

    # Retraining the ML model by eliminate the unfair features
    X_train_eliminate, X_test_eliminate, eliminate_model = retraining_method(X_train, X_test, y_train, D1_select_explain_result, D2_select_explain_result)

    # Evaluating the procedural fairness of the model with the GPF_FAE metric after eliminate unfair features
    D1_select_eliminate, D2_select_eliminate, D1_select_explain_result_eliminate, D2_select_explain_result_eliminate, GPF_FAE_result_eliminate = \
        GPF_FAE_metric(X_train_eliminate, X_test_eliminate, eliminate_model, sensitive_feature_test, n=100)
    print("GPF_FAE Metric After Retraining: ", GPF_FAE_result_eliminate)








