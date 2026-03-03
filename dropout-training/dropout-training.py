import numpy as np

def dropout(x, p=0.5, seed=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """

    # Write code here
    
    rng = np.random.default_rng(seed)
    
    x = np.array(x)

    dropout_pattern = (rng.random(size=x.shape) < 1 - p).astype(int) / (1-p)

    output = x * dropout_pattern
    
    return (output, dropout_pattern)