# core.py

import numpy as np
import operator
from scipy.stats import gaussian_kde, mode
import matplotlib.pyplot as plt
from .constants import DEFAULT_STATISTICS_SAMPLE_SIZE, DEFAULT_PLOTTING_SAMPLE_SIZE



class StochasticVariable:
    def __init__(self, distribution=None, dependencies=None, func=None, name=None, distribution_type=None, value=None):
        """
        Represents a stochastic variable.

        Parameters:
            - distribution: An instance of a distribution (e.g., NormalDistribution).
            - dependencies: A list of other StochasticVariables this variable depends on.
            - func: A callable that generates values based on dependencies.
            - name: Optional name of the stochastic variable.
            - distribution_type: 'continuous', 'discrete', or 'mixed'.
            - value: If the variable represents a constant, this is its value.
        """
        if value is not None and (distribution is not None or func is not None or dependencies):
            raise ValueError("A StochasticVariable cannot have a value alongside distribution, function, or dependencies.")

        self.distribution = distribution
        self.func = func
        self.value = value  # Store the constant value if applicable
        self.name = name or (str(value) if value is not None else f"SV_{id(self)}")

        if dependencies is None:
            dependencies = []

        # Add dependencies from the distribution, if present
        if distribution is not None:
            dependencies.extend(distribution.get_dependencies())
            self.distribution_type = distribution.distribution_type
        else:
            self.distribution_type = distribution_type  # User-specified type, if applicable

        self.dependencies = dependencies

        # Validate for circular dependencies
        self._check_circular_dependency()

        # Determine the type for constants
        if self.value is not None:
            self.distribution_type = (
                'discrete' if isinstance(self.value, int) else 'continuous'
            )

        # Automatically determine distribution type if not specified
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


    def get_all_dependencies(self, visited=None, stack=None):
        """
        Recursively collects all dependencies of the current variable.

        Parameters:
            - visited (set): Tracks visited variables to avoid redundant checks.
            - stack (set): Tracks variables in the current recursion stack to detect cycles.

        Returns:
            - set: A set of all dependencies.
        """
        if visited is None:
            visited = set()
        if stack is None:
            stack = set()

        if self in stack:
            # Circular dependency detected
            raise ValueError(f"Circular dependency detected for variable '{self.name}'.")

        if self in visited:
            # Already processed, no need to recurse further
            return set()

        # Mark as being visited in the current recursion stack
        stack.add(self)
        visited.add(self)

        # Collect dependencies
        deps = set(self.dependencies)
        for dep in self.dependencies:
            deps.update(dep.get_all_dependencies(visited, stack))

        # Remove from recursion stack after processing
        stack.remove(self)

        return deps



    def sample(self, size=1, context=None):
        """
        Generates samples from the stochastic variable.

        Parameters:
            - size (int): Number of samples to generate.
            - context (dict): Tracks dependencies to ensure consistent sampling.

        Returns:
            - numpy.ndarray: Samples of the stochastic variable.
        """
        if context is None:
            context = {}

        if self in context:
            return context[self]

        # Handle constant variables
        if self.value is not None:
            samples = np.full(size, self.value)  # Return constant value
            context[self] = samples
            return samples

        # Sample dependencies first
        dep_samples = [dep.sample(size=size, context=context) for dep in self.dependencies]

        # Apply the function or distribution
        if self.distribution is not None:
            samples = self.distribution.sample(size=size)
        elif self.func is not None:
            samples = self.func(*dep_samples)
        else:
            raise ValueError(f"StochasticVariable '{self.name}' must have a distribution, function, or constant value.")

        context[self] = samples
        return samples



    def pdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, bandwidth='scott'):
        if self.distribution_type == 'discrete':
            raise ValueError(f"PDF is not defined for discrete variable '{self.name}'. Use pmf instead.")
        elif self.distribution_type in ['continuous', 'mixed']:
            if self.distribution is not None and hasattr(self.distribution, 'pdf'):
                return self.distribution.pdf(x)
            else:
                return self.empirical_pdf(x, size=size, bandwidth=bandwidth)
        else:
            raise ValueError(f"Unknown distribution type for variable '{self.name}'.")

    def pmf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        if self.distribution_type == 'continuous':
            raise ValueError(f"PMF is not defined for continuous variable '{self.name}'. Use pdf instead.")
        elif self.distribution_type in ['discrete', 'mixed']:
            if self.distribution is not None and hasattr(self.distribution, 'pmf'):
                return self.distribution.pmf(x)
            else:
                return self.empirical_pmf(x, size=size)
        else:
            raise ValueError(f"Unknown distribution type for variable '{self.name}'.")

    def cdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Computes the cumulative distribution function (CDF) for the variable.

        Parameters:
        - x: The value or array of values at which to compute the CDF.
        - size: Number of samples to use if computing the empirical CDF.

        Returns:
        - float or numpy.ndarray: The CDF value(s) at the specified x.
        """
        if self.distribution is not None and hasattr(self.distribution, 'cdf'):
            # Use the underlying distribution's CDF if available
            return self.distribution.cdf(x)
        else:
            # Compute empirical CDF
            return self.empirical_cdf(x, size=size)


    def empirical_pdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, bandwidth='scott'):
        samples = self.sample(size=size)
        kde = gaussian_kde(samples, bw_method=bandwidth)
        return kde.evaluate(x)

    def empirical_pmf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        counts = np.array([np.sum(samples == xi) for xi in np.atleast_1d(x)])
        return counts / size

    def empirical_cdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Computes the empirical cumulative distribution function (ECDF).

        Parameters:
        - x: The value or array of values at which to compute the ECDF.
        - size: Number of samples to use for the empirical computation.

        Returns:
        - float or numpy.ndarray: The ECDF value(s) at the specified x.
        """
        samples = np.sort(self.sample(size=size))
        x = np.atleast_1d(x)
        ecdf_values = np.searchsorted(samples, x, side='right') / len(samples)
        return ecdf_values if x.ndim > 0 else ecdf_values[0]
    

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
            size = DEFAULT_STATISTICS_SAMPLE_SIZE  # Default sample size for confidence interval
        samples = self.sample(size=size)
        alpha = 1 - confidence_level
        lower_bound = np.percentile(samples, 100 * alpha / 2)
        upper_bound = np.percentile(samples, 100 * (1 - alpha / 2))
        return lower_bound, upper_bound

    def mean(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Calculates the mean of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
        
        Returns:
            - The estimated mean.
        """
        samples = self.sample(size=size)
        return np.mean(samples)

    def std(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Calculates the standard deviation of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
        
        Returns:
            - The estimated standard deviation.
        """
        samples = self.sample(size=size)
        return np.std(samples)

    def var(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Calculates the variance of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
        
        Returns:
            - The estimated variance.
        """
        samples = self.sample(size=size)
        return np.var(samples)

    def median(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Calculates the median of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
        
        Returns:
            - The estimated median.
        """
        samples = self.sample(size=size)
        return np.median(samples)

    def mode(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Calculates the mode of the stochastic variable.
        
        Parameters:
            - size (int): The number of samples to draw for the estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
        
        Returns:
            - The estimated mode.
        """
        samples = self.sample(size=size)
        mode_value, _ = mode(samples, keepdims=True)
        return mode_value[0]

    def nth_moment(self, n, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        """
        Calculates the nth moment of the stochastic variable.
        
        Parameters:
            - n (int): The order of the moment to calculate.
            - size (int): The number of samples to draw for the estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
        
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

    def plot(self, size=DEFAULT_PLOTTING_SAMPLE_SIZE, bins=30, density=True, title=None):
        """
        Plots the distribution of the stochastic variable.

        Parameters:
        - size (int): Number of samples to draw for the plot (default: DEFAULT_PLOTTING_SAMPLE_SIZE).
        - bins (int): Number of bins in the histogram (default: 30).
        - density (bool): Whether to normalize the histogram (default: True).
        - title (str): Title for the plot (optional).
        """
        samples = self.sample(size=size)

        plt.figure(figsize=(8, 6))
        # Histogram of samples
        plt.hist(samples, bins=bins, density=density, alpha=0.6, color='blue', edgecolor='black', label='Histogram')

        # If PDF is available, plot it
        if hasattr(self.distribution, 'pdf'):
            x_range = np.linspace(min(samples), max(samples), DEFAULT_PLOTTING_SAMPLE_SIZE)
            pdf_values = self.distribution.pdf(x_range)
            plt.plot(x_range, pdf_values, color='red', label='PDF')

        # Add labels and legend
        plt.xlabel('Value')
        plt.ylabel('Density' if density else 'Frequency')
        plt.title(title or f"Distribution of {self.name}")
        plt.legend()
        plt.grid(True)
        plt.show()


class StochasticVector:
    def __init__(self, *variables, name=None):
        """
        Initializes a stochastic vector.

        Parameters:
        - variables: StochasticVariable instances to include in the vector.
        - name (str): Name of the stochastic vector (default: None).
        """
        if not all(isinstance(var, StochasticVariable) for var in variables):
            raise ValueError("All inputs must be instances of StochasticVariable.")

        self.variables = variables
        self.name = name or "StochasticVector"

    def sample(self, size=1):
        """
        Samples from all component variables.

        Parameters:
        - size (int): Number of samples to generate.

        Returns:
        - numpy.ndarray: Matrix where each row corresponds to samples from the vector.
        """
        return np.array([var.sample(size=size) for var in self.variables]).T

    def get_all_dependencies(self):
        """
        Collects dependencies from all component variables.

        Returns:
        - set: Combined dependencies of all component variables.
        """
        dependencies = set()
        for var in self.variables:
            # Aggregate all dependencies from each component
            dependencies.update(var.get_all_dependencies())
        # Include direct components as dependencies
        dependencies.update(self.variables)
        return dependencies

    # Norm (||v||)
    def norm(self, p=2):
        """
        Computes the p-norm of the stochastic vector.

        Parameters:
            - p (int or float): Order of the norm (default: 2 for L2-norm).

        Returns:
            - StochasticVariable: The norm of the vector as a new stochastic variable.
        """
        return apply(
            lambda *components: np.linalg.norm(np.column_stack(components), ord=p, axis=1),
            *self.variables,
            name=f"{self.name}_norm_{p}"
        )



    # Dot Product
    def dot(self, other):
        """
        Computes the dot product with another stochastic vector.

        Parameters:
            - other: StochasticVector (must have the same size as self).

        Returns:
            - StochasticVariable: The dot product as a new stochastic variable.
        """
        if not isinstance(other, StochasticVector):
            raise ValueError("Dot product requires another StochasticVector.")
        if len(self.variables) != len(other.variables):
            raise ValueError("Vectors must have the same number of components for dot product.")

        # Use `apply` to compute the dot product
        return apply(
            lambda *components: sum(x * y for x, y in zip(components[:len(self.variables)], components[len(self.variables):])),
            *self.variables,
            *other.variables,
            name=f"dot({self.name}, {other.name})"
        )



    # Cross Product (for 3D vectors only)
    def cross(self, other):
        """
        Computes the cross product with another 3D stochastic vector.

        Parameters:
        - other (StochasticVector): The other vector.

        Returns:
        - StochasticVector: The cross product as a new stochastic vector.
        """
        if not isinstance(other, StochasticVector):
            raise ValueError("Cross product requires another StochasticVector.")
        if len(self.variables) != 3 or len(other.variables) != 3:
            raise ValueError("Cross product is only defined for 3D vectors.")

        def cross_func(x1, y1, z1, x2, y2, z2):
            return [
                y1 * z2 - z1 * y2,
                z1 * x2 - x1 * z2,
                x1 * y2 - y1 * x2
            ]

        components = [
            StochasticVariable(
                dependencies=self.variables + other.variables,
                func=lambda *args, i=i: cross_func(*args)[i],
                name=f"cross({self.name}, {other.name})_{axis}"
            )
            for i, axis in enumerate(["x", "y", "z"])
        ]
        return StochasticVector(*components, name=f"cross({self.name}, {other.name})")


    # Overloaded Operators for Hadamard-Type Operations
    def __add__(self, other):
        return self._hadamard_operation(other, lambda x, y: x + y, "add")

    def __sub__(self, other):
        return self._hadamard_operation(other, lambda x, y: x - y, "sub")

    def __mul__(self, other):
        return self._hadamard_operation(other, lambda x, y: x * y, "mul")

    def __truediv__(self, other):
        return self._hadamard_operation(other, lambda x, y: x / y, "div")

    # Hadamard Operation Helper
    def _hadamard_operation(self, other, operator, op_name):
        """
        Applies an element-wise operation to this stochastic vector and another vector/scalar.

        Parameters:
            - other: StochasticVector, StochasticVariable, or scalar.
            - operator (callable): The operation to apply (e.g., lambda x, y: x + y).
            - op_name (str): Name of the operation (used for naming the result).

        Returns:
            - StochasticVector: The resulting stochastic vector.
        """
        if isinstance(other, StochasticVector):
            if len(self.variables) != len(other.variables):
                raise ValueError(f"Both vectors must have the same number of components for {op_name}.")

            combined_vars = [
                apply(operator, v1, v2, name=f"{op_name}({v1.name}, {v2.name})")
                for v1, v2 in zip(self.variables, other.variables)
            ]
        elif isinstance(other, StochasticVariable) or np.isscalar(other):
            combined_vars = [
                apply(operator, var, other, name=f"{op_name}({var.name}, {other})")
                for var in self.variables
            ]
        else:
            raise ValueError(f"Unsupported operand type for {op_name} operation.")

        return StochasticVector(*combined_vars, name=f"{self.name}_{op_name}")



    # Repr
    def __repr__(self):
        return f"StochasticVector(name={self.name}, variables={[var.name for var in self.variables]})"


# Core functions



def apply(func, *args, name=None):
    """
    Create a new StochasticVariable by applying a function to a list of variables or constants.

    Parameters:
        - func: A callable that operates on the inputs.
        - *args: An arbitrary number of StochasticVariables or constants.
        - name: Optional name for the new StochasticVariable.

    Returns:
        - A new StochasticVariable representing the result of applying func to the inputs.
    """
    dependencies = []
    constant_values = []

    for arg in args:
        if isinstance(arg, StochasticVariable):
            dependencies.append(arg)
        else:
            # Collect constant values separately
            constant_values.append(arg)
            const_var = StochasticVariable(
                func=lambda value=arg: np.array(value),
                name=str(arg),
                distribution_type='discrete' if isinstance(arg, int) else 'continuous'
            )
            dependencies.append(const_var)

    # Define the function to apply to the samples
    def new_func(*samples):
        # Separate constant and stochastic samples
        stochastic_samples = samples[:len(dependencies) - len(constant_values)]
        constants = constant_values
        return func(*stochastic_samples, *constants)

    # Determine the distribution type for the new variable
    distribution_types = {var.distribution_type for var in dependencies}
    if len(distribution_types) == 1:
        distribution_type = distribution_types.pop()
    else:
        distribution_type = 'mixed'

    # Create the new StochasticVariable
    new_var = StochasticVariable(
        dependencies=dependencies,
        func=new_func,
        name=name or f"FunctionVariable_{id(new_func)}",
        distribution_type=distribution_type
    )
    return new_var





def probability(condition, *args, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
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


