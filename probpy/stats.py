# stats.py

import numpy as np

def mean(data):
    """
    Calculate the mean of a dataset.
    
    Parameters:
        data (list or numpy.ndarray): Input dataset.
    
    Returns:
        float: Mean of the dataset.
    """
    return np.mean(data)

def variance(data):
    """
    Calculate the variance of a dataset.
    
    Parameters:
        data (list or numpy.ndarray): Input dataset.
    
    Returns:
        float: Variance of the dataset.
    """
    return np.var(data)
