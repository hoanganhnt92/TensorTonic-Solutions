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
    Xarray = np.asarray(X)

    w = np.zeros(Xarray.shape[1])
    b = 0
    
    for i in range(steps):
        z = X@w + b
        p = _sigmoid(z)
        
        deltaW = Xarray.T @ (p - y)/ Xarray.shape[0]
        deltaB = np.mean(p - y)
        
        w -= deltaW * lr
        b -= deltaB * lr
    
    return (w, b)