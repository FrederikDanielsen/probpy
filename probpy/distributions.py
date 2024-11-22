# distributions.py

import numpy as np
from scipy.stats import bernoulli, norm

def generate_bernoulli(p, size):
    """
    Generate samples from a Bernoulli distribution.
    
    Parameters:
        p (float): Probability of success (0 <= p <= 1).
        size (int): Number of samples to generate.
    
    Returns:
        numpy.ndarray: Array of Bernoulli samples.
    """
    return bernoulli.rvs(p, size=size)

def generate_normal(mean, std, size):
    """
    Generate samples from a Normal distribution.
    
    Parameters:
        mean (float): Mean of the distribution.
        std (float): Standard deviation.
        size (int): Number of samples to generate.
    
    Returns:
        numpy.ndarray: Array of Normal samples.
    """
    return norm.rvs(loc=mean, scale=std, size=size)
