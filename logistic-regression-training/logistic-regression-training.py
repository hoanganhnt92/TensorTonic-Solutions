import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    # correct type
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    
    # init
    [n_samples, n_features] = X_array.shape
    W = np.zeros(n_features)
    b = 0
    
    for i in range(steps):
        # forward
        z = X_array@W + b

        # loss
        p = _sigmoid(z)
        deltaW = X_array.T @ (p - y_array)/ n_samples
        deltaB = np.mean(p - y_array)

        # update
        W -= deltaW * lr
        b -= deltaB * lr
    
    return (W, b)