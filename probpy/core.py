# core.py

# IMPORTS
import numpy as np
import operator
from scipy.stats import gaussian_kde
from .constants import DEFAULT_STATISTICS_SAMPLE_SIZE
from scipy.stats import t, chi2, distributions

# Core classes

class StochasticVariable:

    instances = []

    @classmethod
    def delete_all_instances(cls):
        # Delete all instances by iterating over the tracked instances

        while cls.instances:
            instance = cls.instances.pop() 
            del instance

        
    def __init__(
        self,
        distribution=None,
        dependencies=None,
        func=None,
        name=None,
        distribution_type=None,
        value=None,
    ):
        """
        Represents a stochastic variable.

        Parameters:
            - distribution: An instance of a distribution.
            - dependencies: A list of other StochasticVariables this variable depends on.
            - func: A callable that generates values based on dependencies.
            - name: Optional name of the stochastic variable.
            - distribution_type: 'continuous', 'discrete', or 'mixed'.
            - value: If the variable represents a constant, this is its value.
        """
        if (
            value is not None
            and (distribution is not None or func is not None or dependencies)
        ):
            raise ValueError(
                "A StochasticVariable cannot have a value alongside distribution, function, or dependencies."
            )

        self.distribution = distribution
        self.func = func
        self.value = value  # Store the constant value if applicable

        for instance in StochasticVariable.instances:
            if instance.name == name:
                raise ValueError(f'A stochastic variable with the name "{name}" already exists!')

        self.name = name or (str(value) if value is not None else f"SV_{id(self)}")
        StochasticVariable.instances.append(self)

        if dependencies is None:
            dependencies = []

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
                "discrete" if isinstance(self.value, int) else "continuous"
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
            return "mixed"

    def _check_circular_dependency(self):
        all_deps = self.get_all_dependencies()
        if self in all_deps:
            raise ValueError(
                f"Circular dependency detected for variable '{self.name}'."
            )

    def get_all_dependencies(self, visited=None, stack=None):
        if visited is None:
            visited = set()
        if stack is None:
            stack = set()

        if self in stack:
            raise ValueError(
                f"Circular dependency detected for variable '{self.name}'."
            )

        if self in visited:
            return set()

        stack.add(self)
        visited.add(self)

        deps = set(self.dependencies)
        for dep in self.dependencies:
            deps.update(dep.get_all_dependencies(visited, stack))

        stack.remove(self)

        return deps

    def sample(self, size=1, context=None):
        if context is None:
            context = {}

        if self in context:
            cached_samples = context[self]
            if len(cached_samples) >= size:
                return cached_samples[:size]
            else:
                additional_samples = self._generate_samples(
                    size - len(cached_samples), context
                )
                context[self] = np.concatenate([cached_samples, additional_samples])
                return context[self][:size]

        samples = self._generate_samples(size, context)
        context[self] = samples
        return samples if len(samples) > 1 else samples[0]

    def _generate_samples(self, size, context):
        if self.value is not None:
            return np.full(size, self.value)

        dep_samples = [
            dep.sample(size=size, context=context) for dep in self.dependencies
        ]

        if self.distribution is not None:
            return self.distribution.sample(size=size, context=context)
        elif self.func is not None:
            return self.func(*dep_samples)
        else:
            raise ValueError(
                f"StochasticVariable '{self.name}' must have a distribution, function, or constant value."
            )

    def pdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, bandwidth='scott', context=None):
        if self.distribution_type == 'discrete':
            raise ValueError(
                f"PDF is not defined for discrete variable '{self.name}'. Use pmf instead."
            )
        elif self.distribution_type in ['continuous', 'mixed']:
            if self.distribution is not None and hasattr(self.distribution, 'pdf'):
                return self.distribution.pdf(x, context=context)
            else:
                return self.empirical_pdf(x, size=size, bandwidth=bandwidth)
        else:
            raise ValueError(
                f"Unknown distribution type for variable '{self.name}'."
            )

    def pmf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, context=None):
        if self.distribution_type == 'continuous':
            raise ValueError(
                f"PMF is not defined for continuous variable '{self.name}'. Use pdf instead."
            )
        elif self.distribution_type in ['discrete', 'mixed']:
            if self.distribution is not None and hasattr(self.distribution, 'pmf'):
                return self.distribution.pmf(x, context=context)
            else:
                return self.empirical_pmf(x, size=size)
        else:
            raise ValueError(
                f"Unknown distribution type for variable '{self.name}'."
            )

    def cdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, context=None):
        if self.distribution is not None and hasattr(self.distribution, 'cdf'):
            return self.distribution.cdf(x, context=context)
        else:
            return self.empirical_cdf(x, size=size)

    def empirical_pdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, bandwidth='scott'):
        if self.distribution_type in ['continuous', 'mixed']:
            samples = self.sample(size=size)
            kde = gaussian_kde(samples, bw_method=bandwidth)
            return kde.evaluate(x)
        else:
            raise ValueError(
                f"Empirical PDF is not defined for discrete variable '{self.name}'."
            )

    def empirical_pmf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        if self.distribution_type in ['discrete', 'mixed']:
            samples = self.sample(size=size)
            counts = np.array([np.sum(samples == xi) for xi in np.atleast_1d(x)])
            return counts / size
        else:
            raise ValueError(
                f"Empirical PMF is not defined for continuous variable '{self.name}'."
            )

    def empirical_cdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = np.sort(self.sample(size=size))
        x = np.atleast_1d(x)
        ecdf_values = np.searchsorted(samples, x, side='right') / len(samples)
        return ecdf_values if x.ndim > 0 else ecdf_values[0]

    # Statistical methods
    def mean(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        return np.mean(samples)

    def std(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        return np.std(samples, ddof=1)

    def var(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        return np.var(samples, ddof=1)

    def median(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        return np.median(samples)

    def mode(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        if self.distribution_type == 'continuous':
            kde = gaussian_kde(samples)
            x_grid = np.linspace(np.min(samples), np.max(samples), 1000)
            pdf_values = kde.evaluate(x_grid)
            mode_index = np.argmax(pdf_values)
            return x_grid[mode_index]
        elif self.distribution_type == 'discrete':
            values, counts = np.unique(samples, return_counts=True)
            max_count_indices = np.where(counts == np.max(counts))[0]
            return values[max_count_indices[0]]  # Return the first mode
        else:
            raise ValueError("Mode calculation is not defined for mixed distributions.")

    def nth_moment(self, n, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        return np.mean(samples ** n)

    def mean_confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        mean = np.mean(samples)
        sem = np.std(samples, ddof=1) / np.sqrt(size)
        h = sem * t.ppf((1 + confidence_level) / 2., size - 1)
        if self.distribution_type == "discrete":
            return np.floor(mean-h), np.ceil(mean+h)
        return float(mean - h), float(mean + h)
    
    def variance_confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        n = size
        sample_variance = np.var(samples, ddof=1)  # Sample variance with Bessel's correction
        alpha = 1 - confidence_level
        
        # Compute critical chi-squared values
        chi2_lower = chi2.ppf(alpha / 2, df=n-1)  # Lower critical value
        chi2_upper = chi2.ppf(1 - alpha / 2, df=n-1)  # Upper critical value
        
        # Calculate confidence interval
        lower_bound = (n - 1) * sample_variance / chi2_upper
        upper_bound = (n - 1) * sample_variance / chi2_lower
        
        if self.distribution_type == "discrete":
            return np.floor(lower_bound), np.ceil(upper_bound)
        return float(lower_bound), float(upper_bound)

    def confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)

        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)  # e.g., 2.5% for 95% confidence
        upper_percentile = 100 * (1 - alpha / 2)  # e.g., 97.5% for 95% confidence
        
        lower_bound = np.percentile(samples, lower_percentile)
        upper_bound = np.percentile(samples, upper_percentile)
        
        return float(lower_bound), float(upper_bound)


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

    def sample(self, size=1, context=None):
        """
        Samples from all component variables.

        Parameters:
        - size (int): Number of samples to generate.
        - context (dict): Optional context dictionary for sample caching.

        Returns:
        - numpy.ndarray: Matrix where each row corresponds to samples from the vector.
        """
        if context is None:
            context = {}
        samples = [var.sample(size=size, context=context) for var in self.variables]
        return np.column_stack(samples)

    def get_all_dependencies(self, visited=None, stack=None):
        """
        Collects dependencies from all component variables.

        Returns:
        - set: Combined dependencies of all component variables.
        """
        if visited is None:
            visited = set()
        if stack is None:
            stack = set()

        dependencies = set()
        for var in self.variables:
            dependencies.update(var.get_all_dependencies(visited=visited, stack=stack))
            dependencies.add(var)
        return dependencies

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

        return apply(
            lambda *components: np.sum(
                np.array(components[:len(self.variables)]) * np.array(components[len(self.variables):]), axis=0
            ),
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

        # Components of self
        x1, y1, z1 = self.variables
        # Components of other
        x2, y2, z2 = other.variables

        # Compute each component of the cross product using apply
        cross_x = apply(
            lambda y1, z1, y2, z2: y1 * z2 - z1 * y2,
            y1, z1, y2, z2,
            name=f"cross({self.name}, {other.name})_x"
        )
        cross_y = apply(
            lambda z1, x1, z2, x2: z1 * x2 - x1 * z2,
            z1, x1, z2, x2,
            name=f"cross({self.name}, {other.name})_y"
        )
        cross_z = apply(
            lambda x1, y1, x2, y2: x1 * y2 - y1 * x2,
            x1, y1, x2, y2,
            name=f"cross({self.name}, {other.name})_z"
        )

        return StochasticVector(cross_x, cross_y, cross_z, name=f"cross({self.name}, {other.name})")

    # Overloaded Operators for Element-wise Operations
    def __add__(self, other):
        return self._elementwise_operation(other, operator.add, "add")

    def __sub__(self, other):
        return self._elementwise_operation(other, operator.sub, "sub")

    def __mul__(self, other):
        return self._elementwise_operation(other, operator.mul, "mul")

    def __truediv__(self, other):
        return self._elementwise_operation(other, operator.truediv, "div")

    # Element-wise Operation Helper
    def _elementwise_operation(self, other, operator_func, op_name):
        """
        Applies an element-wise operation to this stochastic vector and another vector/scalar.

        Parameters:
            - other: StochasticVector, StochasticVariable, or scalar.
            - operator_func (callable): The operation to apply (e.g., operator.add).
            - op_name (str): Name of the operation (used for naming the result).

        Returns:
            - StochasticVector: The resulting stochastic vector.
        """
        if isinstance(other, StochasticVector):
            if len(self.variables) != len(other.variables):
                raise ValueError(f"Both vectors must have the same number of components for {op_name}.")

            combined_vars = [
                apply(operator_func, v1, v2, name=f"{v1.name}_{op_name}_{v2.name}")
                for v1, v2 in zip(self.variables, other.variables)
            ]
        elif isinstance(other, StochasticVariable) or np.isscalar(other):
            combined_vars = [
                apply(operator_func, var, other, name=f"{var.name}_{op_name}_{other}")
                for var in self.variables
            ]
        else:
            raise ValueError(f"Unsupported operand type for {op_name} operation.")

        return StochasticVector(*combined_vars, name=f"{self.name}_{op_name}")

    # Representation
    def __repr__(self):
        return f"StochasticVector(name={self.name}, variables={[var.name for var in self.variables]})"


def delete(object):
    """
    Deletes StochasticVariable or StochasticVector instance.
    If object is a StochasticVector all StochasticVariables in te vector are deleted.

    Input parameters:
    object: (StochasticVariable or StochasticVector)
    """
    if isinstance(object, StochasticVariable):
        if object in StochasticVariable.instances:
            StochasticVariable.instances.remove(object) 
            del object
        return
    elif isinstance(object, StochasticVector):
        for variable in object.variables:
            delete(variable)
        del object


# Core functions

def apply(func, *args, name=None):
    dependencies = [
        arg if isinstance(arg, StochasticVariable) else StochasticVariable(value=arg)
        for arg in args
    ]

    def new_func(*samples):
        return func(*samples)

    distribution_types = {dep.distribution_type for dep in dependencies if dep.distribution_type}
    distribution_type = distribution_types.pop() if len(distribution_types) == 1 else "mixed"

    return StochasticVariable(
        dependencies=dependencies,
        func=new_func,
        name=name,
        distribution_type=distribution_type,
    )


def probability(condition, *args, size=DEFAULT_STATISTICS_SAMPLE_SIZE, context=None):
    """
    Estimate the probability that a condition involving stochastic variables is True.

    Parameters:
    - condition: A callable that accepts N inputs and returns a boolean array.
    - *args: An arbitrary number of StochasticVariables or constants.
    - size: The number of samples to generate for the estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
    - context: Optional context dictionary to cache samples.

    Returns:
    - Estimated probability as a float between 0 and 1.
    """
    if context is None:
        context = {}

    # Collect samples from args
    samples_list = []
    for arg in args:
        if isinstance(arg, StochasticVariable):
            samples = arg.sample(size=size, context=context)
        else:
            # If arg is a constant, create an array of the constant
            samples = np.full(size, arg)
        samples_list.append(samples)

    # Convert the list of samples into arrays
    samples_tuple = tuple(samples_list)

    # Evaluate the condition across all samples
    condition_results = condition(*samples_tuple)

    # Ensure the condition results are a boolean array
    condition_results = np.asarray(condition_results, dtype=bool)

    # Calculate the probability
    prob = np.mean(condition_results)

    return prob

def set_random_seed(seed):
    """
    Set the random seed for reproducibility in numpy and scipy.
    """
    np.random.seed(seed)
    # Optionally, also configure scipy.stats if applicable
    distributions.rng = np.random.default_rng(seed)
