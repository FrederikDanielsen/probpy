# core.py

import numpy as np
import operator
from scipy.stats import gaussian_kde, mode




class StochasticVariable:
    def __init__(self, distribution=None, dependencies=None, func=None, name=None, distribution_type=None):
        self.distribution = distribution
        self.func = func
        self.name = name or f"SV_{id(self)}"

        if dependencies is None:
            dependencies = []

        if distribution is not None:
            dependencies.extend(distribution.get_dependencies())
            self.distribution_type = distribution.distribution_type
        else:
            self.distribution_type = distribution_type  # User can specify this if needed

        self.dependencies = dependencies
        self._check_circular_dependency()

        if self.distribution_type is None:
            self.distribution_type = self._determine_distribution_type()

    def _determine_distribution_type(self):
        types = set(dep.distribution_type for dep in self.dependencies)
        if len(types) == 1:
            # All dependencies have the same type
            return types.pop()
        else:
            # Mixed types among dependencies
            return 'mixed'

    def _check_circular_dependency(self):
        all_deps = self.get_all_dependencies()
        if self in all_deps:
            raise ValueError(f"Circular dependency detected for variable '{self.name}'.")

    def get_all_dependencies(self, visited=None):
        if visited is None:
            visited = set()
        if self in visited:
            return {self}
        visited.add(self)
        deps = set(self.dependencies)
        for dep in self.dependencies:
            deps.update(dep.get_all_dependencies(visited))
        return deps

    def sample(self, size=1, context=None):
        if context is None:
            context = {}
        if self in context:
            return context[self]
        # Sample dependencies first
        dep_samples = [dep.sample(size=size, context=context) for dep in self.dependencies]
        # Now sample self
        if self.distribution is not None:
            samples = self.distribution.sample(size=size, context=context)
        elif self.func is not None:
            samples = self.func(*dep_samples)
        else:
            raise ValueError(f"StochasticVariable '{self.name}' must have a distribution or function.")
        context[self] = samples
        return samples

    def pdf(self, x, size=10000, bandwidth='scott'):
        if self.distribution_type == 'discrete':
            raise ValueError(f"PDF is not defined for discrete variable '{self.name}'. Use pmf instead.")
        elif self.distribution_type in ['continuous', 'mixed']:
            if self.distribution is not None and hasattr(self.distribution, 'pdf'):
                return self.distribution.pdf(x)
            else:
                return self.empirical_pdf(x, size=size, bandwidth=bandwidth)
        else:
            raise ValueError(f"Unknown distribution type for variable '{self.name}'.")

    def pmf(self, x, size=10000):
        if self.distribution_type == 'continuous':
            raise ValueError(f"PMF is not defined for continuous variable '{self.name}'. Use pdf instead.")
        elif self.distribution_type in ['discrete', 'mixed']:
            if self.distribution is not None and hasattr(self.distribution, 'pmf'):
                return self.distribution.pmf(x)
            else:
                return self.empirical_pmf(x, size=size)
        else:
            raise ValueError(f"Unknown distribution type for variable '{self.name}'.")

    def empirical_pdf(self, x, size=10000, bandwidth='scott'):
        samples = self.sample(size=size)
        kde = gaussian_kde(samples, bw_method=bandwidth)
        return kde.evaluate(x)

    def empirical_pmf(self, x, size=10000):
        samples = self.sample(size=size)
        counts = np.array([np.sum(samples == xi) for xi in np.atleast_1d(x)])
        return counts / size
    

    def confidence_interval(self, confidence_level=0.95, size=None):
        """
        Calculates the confidence interval for the stochastic variable.
        
        Parameters:
            - confidence_level (float): The confidence level for the interval (default: 0.95).
            - size (int): The number of samples to draw for the estimation (default: None, uses class-level default).
        
        Returns:
            - A tuple (lower_bound, upper_bound) representing the confidence interval.
        """
        if size is None:
            size = 10000  # Default sample size for confidence interval
        samples = self.sample(size=size)
        alpha = 1 - confidence_level
        lower_bound = np.percentile(samples, 100 * alpha / 2)
        upper_bound = np.percentile(samples, 100 * (1 - alpha / 2))
        return lower_bound, upper_bound

    def mean(self, size=10000):
        """
        Calculates the mean of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: 10000).
        
        Returns:
            - The estimated mean.
        """
        samples = self.sample(size=size)
        return np.mean(samples)

    def std(self, size=10000):
        """
        Calculates the standard deviation of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: 10000).
        
        Returns:
            - The estimated standard deviation.
        """
        samples = self.sample(size=size)
        return np.std(samples)

    def var(self, size=10000):
        """
        Calculates the variance of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: 10000).
        
        Returns:
            - The estimated variance.
        """
        samples = self.sample(size=size)
        return np.var(samples)

    def median(self, size=10000):
        """
        Calculates the median of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: 10000).
        
        Returns:
            - The estimated median.
        """
        samples = self.sample(size=size)
        return np.median(samples)

    def mode(self, size=10000):
        """
        Calculates the mode of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: 10000).
        
        Returns:
            - The estimated mode.
        """
        samples = self.sample(size=size)
        mode_value, _ = mode(samples, keepdims=True)
        return mode_value[0]

    def nth_moment(self, n, size=10000):
        """
        Calculates the nth moment of the stochastic variable.
        
        Parameters:
            - n (int): The order of the moment to calculate.
            - size (int): The number of samples to draw for the estimation (default: 10000).
        
        Returns:
            - The estimated nth moment.
        """
        samples = self.sample(size=size)
        return np.mean(samples**n)

    # Overloaded arithmetic operators
    def __add__(self, other):
        return apply(operator.add, self, other, name=f"({self.name} + {other})")

    def __radd__(self, other):
        return apply(operator.add, other, self, name=f"({other} + {self.name})")

    def __sub__(self, other):
        return apply(operator.sub, self, other, name=f"({self.name} - {other})")

    def __rsub__(self, other):
        return apply(operator.sub, other, self, name=f"({other} - {self.name})")

    def __mul__(self, other):
        return apply(operator.mul, self, other, name=f"({self.name} * {other})")

    def __rmul__(self, other):
        return apply(operator.mul, other, self, name=f"({other} * {self.name})")

    def __truediv__(self, other):
        return apply(operator.truediv, self, other, name=f"({self.name} / {other})")

    def __rtruediv__(self, other):
        return apply(operator.truediv, other, self, name=f"({other} / {self.name})")

    def __pow__(self, other):
        return apply(operator.pow, self, other, name=f"({self.name} ** {other})")

    def __rpow__(self, other):
        return apply(operator.pow, other, self, name=f"({other} ** {self.name})")
    
    def given(self, **conditions):
        """
        Conditions the stochastic variable on one or more dependencies.

        Parameters:
        - conditions (dict): A mapping of variable names to fixed values.

        Returns:
        - A new StochasticVariable representing the conditional variable.
        """
        # Check if any dependencies are in the conditions
        updated_dependencies = []
        for dep in self.dependencies:
            if dep.name in conditions:
                # Replace the dependency with its conditioned version
                fixed_value = conditions[dep.name]
                updated_dependencies.append(
                    StochasticVariable(
                        distribution=None,
                        func=lambda *args: fixed_value,
                        name=f"{dep.name}={fixed_value}",
                    )
                )
            else:
                # Keep the original dependency
                updated_dependencies.append(dep)

        # Create a new variable with updated dependencies
        return StochasticVariable(
            distribution=self.distribution,
            dependencies=updated_dependencies,
            func=self.func,
            name=f"{self.name}|{','.join([f'{k}={v}' for k, v in conditions.items()])}",
        )



# Core functions




def apply(func, *args, name=None):
    """
    Create a new StochasticVariable by applying a function to a list of variables or constants.

    Parameters:
        - func: A callable (e.g., a binary operator or custom function) that operates on the inputs.
        - *args: An arbitrary number of StochasticVariables or constants.
        - name: Optional name for the new StochasticVariable.

    Returns:
        - A new StochasticVariable representing the result of applying func to the inputs.
    """
    dependencies = []
    wrapped_args = []
    distribution_types = set()

    for arg in args:
        if isinstance(arg, StochasticVariable):
            dependencies.append(arg)
            wrapped_args.append(arg)
            distribution_types.add(arg.distribution_type)
        else:
            # Wrap constants into StochasticVariables
            const_var = StochasticVariable(
                func=lambda value=arg: np.array(value),
                name=str(arg),
                distribution_type='discrete' if isinstance(arg, int) else 'continuous'
            )
            dependencies.append(const_var)
            wrapped_args.append(const_var)
            distribution_types.add(const_var.distribution_type)

    # Determine the distribution type for the new variable
    if len(distribution_types) == 1:
        distribution_type = distribution_types.pop()
    else:
        distribution_type = 'mixed'

    # Define the function to apply to the samples
    def new_func(*samples):
        return func(*samples)

    # Create the new StochasticVariable
    new_var = StochasticVariable(
        dependencies=dependencies,
        func=new_func,
        name=name or f"FunctionVariable_{id(new_func)}",
        distribution_type=distribution_type
    )
    return new_var



def probability(condition, *args, size=10000):
    """
    Estimate the probability that a condition involving stochastic variables is True.

    Parameters:
    - condition: A callable that accepts N inputs and returns True or False.
    - *args: An arbitrary number of StochasticVariables or constants.
    - size: The number of samples to generate for the estimation (default: 10,000).

    Returns:
    - Estimated probability as a float between 0 and 1.
    """
    samples_list = []
    for arg in args:
        if isinstance(arg, StochasticVariable):
            samples = arg.sample(size=size)
            samples_list.append(samples)
        else:
            # If arg is a constant, create an array of the constant
            samples = np.full(size, arg)
            samples_list.append(samples)

    # Convert the list of samples into a tuple of arrays
    samples_tuple = tuple(samples_list)

    # Evaluate the condition across all samples
    condition_results = condition(*samples_tuple)

    # Ensure the condition results are a boolean array
    if not isinstance(condition_results, np.ndarray):
        condition_results = np.array(condition_results)

    # Calculate the probability
    prob = np.mean(condition_results)

    return prob


