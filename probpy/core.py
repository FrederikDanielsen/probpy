# core.py

import numpy as np
from .distributions import StochasticVariable, Distribution


# Core functions


def apply(*Xs, transformation, name="T"):
    """
    Apply a transformation to a stochastic variable.

    Parameters:
        X (StochasticVariable): The stochastic variable to which the transformation is applied.
        transformation (callable): A function implementing the transformation.
        name (str): The name of the transformation. The resulting stochastic variable will have name "name(X.name)"

    Returns:
        StochasticVariable: The transformed X as a new Stochastic variable.
    """
    

    class DummyDistribution(Distribution):
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value
        

    if len(Xs) == 1:
    
        X = Xs[0]

        class TransformedDistribution(Distribution):
            def __init__(self, dist, transformation):
                self.dist = dist
                self.transformation = transformation

            def sample(self, size=1):
                if size == 1:
                    return transformation(self.dist.sample())
                return [transformation(sample) for sample in self.dist.sample(size=size)]


        return StochasticVariable(
            TransformedDistribution(X.distribution if isinstance(X, StochasticVariable) else DummyDistribution(X), transformation),
            name=f"{name}({X.name})",
        )
    
    else:

        class TransformedDistribution(Distribution):
            def __init__(self, *dists, transformation):
                self.dists = dists
                self.transformation = transformation

            def sample(self, size=1):
                if size == 1:
                    return transformation(*[dist.sample() for dist in self.dists])
                return [transformation(*[dist.sample() for dist in self.dists]) for _ in range(size)]


        return StochasticVariable(
            TransformedDistribution(*[X.distribution if isinstance(X, StochasticVariable) else DummyDistribution(X) for X in Xs], transformation=transformation),
            name=f"{name}{(X.name for X in Xs)}",
        )
        


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