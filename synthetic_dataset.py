import numpy as np

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def generate_synthetic_data(s0, w1, w2, w3, w4):
    np.random.seed(42)

    # Constructe each feature
    x1 = np.random.normal(0, 1, 10000)
    x2 = np.random.normal(0, 1, 10000)
    z = np.concatenate((np.ones(6000), np.ones(4000) * 0))
    x_proxy = np.concatenate((np.random.normal(1, 0.1, 6000), np.random.normal(0, 0.1, 4000)))
    X = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1), z.reshape(-1, 1), x_proxy.reshape(-1, 1)), axis=1)
    sensitive_feature_idx = 2

    # Construct the label
    s = s0 + w1 * x1 + w2 * x2 + w3 * x_proxy + w4 * z + np.random.normal(0, 1, 10000)
    y = sigmoid(s)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    return X, sensitive_feature_idx, y