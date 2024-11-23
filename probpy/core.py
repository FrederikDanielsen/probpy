# core.py

import numpy as np

# Core constants

DEFAULT_SAMPLE_SIZE = 10000


# Core functions

def probability(*stochastic_variables, condition, size=None):
    """
    Calculate the empirical probability of an event based on samples from multiple stochastic variables.

    Parameters:
        *stochastic_variables (StochasticVariable): One or more stochastic variables to evaluate.
        condition (callable): A function that takes multiple samples (one from each stochastic variable)
                              and returns True if the condition is satisfied, False otherwise.
        size (int): Number of samples to use for the probability calculation.
                    Defaults to the `statistic_sample_size` of the first stochastic variable.

    Returns:
        float: The empirical probability of the event.
    """
    if len(stochastic_variables) == 0:
        raise ValueError("At least one stochastic variable must be provided.")

    # Determine sample size
    if size is None:
        size = stochastic_variables[0].statistic_sample_size

    # Generate samples for each stochastic variable
    samples = [var.sample(size=size) for var in stochastic_variables]

    # Transpose the samples to align them by index
    # For example: [[x1, x2], [y1, y2]] -> [(x1, y1), (x2, y2)]
    samples_transposed = zip(*samples)

    # Apply the condition function to each set of samples and calculate the fraction of True values
    satisfied = np.array([condition(*sample_set) for sample_set in samples_transposed])
    return np.mean(satisfied)