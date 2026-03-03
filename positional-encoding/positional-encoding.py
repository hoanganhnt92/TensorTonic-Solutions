import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    positions = np.arange(seq_len).reshape(seq_len, 1)
    evens = np.arange(0, d_model, 2)

    angles = positions / base ** (evens / d_model)
    
    PE = np.zeros((seq_len, d_model))
    PE[:, 0::2] = np.sin(angles)
    
    odds = angles[:, :PE[:, 1::2].shape[1]]
    PE[:, 1::2] = np.cos(odds)

    return PE