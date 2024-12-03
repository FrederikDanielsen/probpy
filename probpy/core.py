# core.py

__all__ = ["StochasticVariable", "StochasticVector", "apply", "probability", "set_random_seed", "delete"]

# IMPORTS
import numpy as np
import operator
from scipy.stats import gaussian_kde
from .constants import DEFAULT_STATISTICS_SAMPLE_SIZE
from scipy.stats import t, chi2, distributions
from numbers import Number
from functools import reduce



# Core classes


class StochasticVariable:

    instances = []

    @classmethod
    def delete_all_instances(cls):
        # Delete all instances by iterating over the tracked instances

        while cls.instances:
            instance = cls.instances.pop() 
            del instance


    def __init__(self, distribution=None, name=None):
        """
        Represents a stochastic variable.

        Parameters:
            - distribution: An instance of a Distribution.
            - name: Optional name of the stochastic variable.
        """

        if not distribution:
            from .distributions import ContinuousUniformDistribution
            distribution = ContinuousUniformDistribution(0, 1)

        #if any(instance.name == name and not instance.distribution_type == "constant" for instance in StochasticVariable.instances):
        #    raise ValueError(f'A stochastic variable with the name "{name}" already exists!')


        self.distribution = distribution
        self.name = name or f"SV_{id(self)}"

        self._check_circular_dependency()
        StochasticVariable.instances.append(self)

    @property
    def in_use(self):
        for var in StochasticVariable.instances:
            if self in var.get_all_dependencies():
                return True
        return False


    @property
    def distribution_type(self):
        return self.distribution.distribution_type


    @property
    def dependencies(self):
        return self.distribution.get_dependencies()

    def set_distribution(self, dist):
        from .distributions import Distribution
        if not isinstance(dist, Distribution):
            raise ValueError("Input must be of type Distribution!")
        if self.in_use:
            raise Exception("Cannot set distribution of variable that is in use!")
        self.distribution = dist
        

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

        deps = self.distribution.get_dependencies()
        for dep in deps:
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
                additional_samples = self.distribution.sample(size - len(cached_samples), context=context)
                context[self] = np.concatenate([cached_samples, additional_samples])
                return context[self][:size]

        samples = self.distribution.sample(size=size, context=context)
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
        return self.distribution.mean(size=size)

    def std(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.std(size=size)

    def var(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.var(size=size)
    
    def median(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.median(size=size)

    def mode(self, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.mode(size=size)

    def nth_moment(self, n, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.nth_moment(n, size=size)

    def mean_confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.mean_confidence_interval(confidence_level=confidence_level, size=size)
    
    def variance_confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.variance_confidence_interval(confidence_level=confidence_level, size=size)

    def confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        return self.distribution.confidence_interval(confidence_level=confidence_level, size=size)

    def summary(self):
        print("-"*60)
        print("Summary of", self.name)
        self.distribution.summary()

    def print(self):
        print(self)

    def __str__(self):
        from .distributions import ConstantDistribution
        if isinstance(self.distribution, ConstantDistribution):
            return str(self.distribution.value)
        return self.name

    # Overloaded arithmetic operators

    def __add__(self, other):
        if other == 0:
            return self
        else:
            return apply(operator.add, self, other, name=operation_name("+", self, other))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return apply(operator.add, other, self, name=operation_name("+", other, self))

    def __sub__(self, other):
        if other == 0:
            return self
        else:
            return apply(operator.sub, self, other, name=operation_name("-", self, other))

    def __rsub__(self, other):
        if other == 0:
            return self
        else:
            return apply(operator.sub, other, self, name=operation_name("-", other, self))

    def __mul__(self, other):
        if other == 1:
            return self
        else:
            return apply(operator.mul, self, other, name=operation_name("*", self, other))

    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            return apply(operator.mul, other, self, name=operation_name("*", other, self))

    def __truediv__(self, other):
        if other == 1:
            return self
        else:
            return apply(operator.truediv, self, other, name=operation_name("/", self, other))

    def __rtruediv__(self, other):
        return apply(operator.truediv, other, self, name=operation_name("/", other, self))

    def __pow__(self, other):
        if other == 0:
            return 1
        elif other == 1:
            return self
        else:
            return apply(operator.pow, self, other, name=operation_name("^", self, other))

    def __rpow__(self, other):
        if other == 0:
            return 0
        else:
            return apply(operator.pow, other, self, name=operation_name("^", other, self))


class StochasticVector:

    def __init__(self, *variables, name=None):
        """
        Initializes a stochastic vector.

        Parameters:
        - variables: StochasticVariable instances to include in the vector.
        - name (str): Name of the stochastic vector (default: None).
        """

        self.variables = []

        if len(variables) == 1 and isinstance(variables[0], list): 
            variables = list(variables)[0]
        else:
            variables = list(variables)

        for var in variables:
            self.append(var)

        self.name = name or "SVec_" + str(id(self))

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
        return np.column_stack(samples)[0].tolist() if np.shape(np.column_stack(samples))[0] == 1 else np.column_stack(samples).tolist()

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

    def append(self, element):
        if isinstance(element, Number):
            from .distributions import ConstantDistribution
            self.variables.append(StochasticVariable(ConstantDistribution(element)))
        elif isinstance(element, StochasticVariable):
             self.variables.append(element)
        else:
            raise ValueError(f"Element must be of type 'int', 'float', or 'StochasticVariable'. Got '{type(element)}'") 
        
    def insert(self, index, element):
        if isinstance(element, Number):
            from .distributions import ConstantDistribution
            self.variables.insert(index, StochasticVariable(ConstantDistribution(element)))
        elif isinstance(element, StochasticVariable):
             self.variables.insert(index, element)
        else:
            raise ValueError(f"Element must be of type 'int', 'float', or 'StochasticVariable'. Got '{type(element)}'")   

    def pop(self, identifier=None):
        if identifier is None:
            identifier = len(self) - 1        
        elif isinstance(identifier, int):
            return self.variables.pop(identifier)
        else:
            raise ValueError(f"Identifier must be of type 'int' to indicate position in vector or 'str' to indicate the name of the element. Got {type(identifier)}!")

    def remove(self, identifier):   
        if isinstance(identifier, int):
            self.variables.pop(identifier)
        elif isinstance(identifier, StochasticVariable):
            if identifier in self.variables:
                self.variables.remove(identifier)
            else:
                raise ValueError(f"No variable in vector with name '{identifier.name}'!")
        else:
            raise ValueError(f"Identifier must be of type 'int' to indicate position in vector or 'str' to indicate the name of the element. Got {type(identifier)}!")

    def print(self):
        print(self)

    def length(self):
        return len(self)
    
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
            name=f"({self.name})_norm_{p}"
        )
  
    def dot(self, other):  # Dot Product
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

    def cross(self, other): # Cross Product (for 3D vectors only)
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

    def __str__(self):
        n = len(self)
        string = ""
        string += "["
        for i, element in enumerate(self.variables):
            string += str(element)
            if i < n-1:
                string += ", "
        string += "]"
        return string

    def __getitem__(self, identifier):
        if isinstance(identifier, int):
            return self.variables[identifier]
        else:
            raise ValueError(f"Identifier must be of type 'int' to indicate position in vector or 'str' to indicate the name of the element. Got {type(identifier)}!")

    def __setitem__(self, index, element):
        if isinstance(index, int):
            if isinstance(element, Number):
                self.variable[index] = (StochasticVariable(value=element, name=f"_ProbPy_Constant({element})"))
            elif isinstance(element, StochasticVariable):
                self.variables[index] = element
            else:
                raise ValueError(f"Element must be of type int, float, or StochasticVariable. Got '{type(element)}'") 
        else:
            raise ValueError(f"Index must be of type 'int' to indicate position in vector. Got {type(index)}!")            

    def __len__(self):
        return len(self.variables) 


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

        combined_vars = []

        if isinstance(other, StochasticVector):
            if len(self.variables) != len(other.variables):
                raise ValueError(f"Both vectors must have the same number of components for {op_name}.")

            for var, other_var in zip(self.variables, other.variables):
                combined_vars.append(apply(operator_func, var, other_var, name=operation_name(op_name, var, other_var)))

        elif isinstance(other, StochasticVariable) or np.isscalar(other):            
            for var in self.variables:
                    combined_vars.append(apply(operator_func, var, other, name=operation_name(op_name, var, other)))
        else:
            raise ValueError(f"Unsupported operand type for {op_name} operation.")

        return StochasticVector(*combined_vars, name=f"{str(self)}_{op_name}_{str(other)}")

    # Representation
    def __repr__(self):
        return f"StochasticVector(name={self.name}, variables={[var.name for var in self.variables]})"


class StochasticMatrix:

    def __init__(self, matrix, name=None):
        """
        Initializes a stochastic matrix.

        Parameters:
            - matrix: A 2D list or numpy array of StochasticVariable instances.
            - name (str): Name of the stochastic matrix (default: None).
        """

        for i, element in enumerate(matrix):
            if isinstance(element, StochasticVector):
                matrix[i] = element.variables

        self.matrix = np.array(matrix, dtype=object)
        for i, element in enumerate(self.matrix.flatten()):
            if not isinstance(element, StochasticVariable):
                if isinstance(element, Number):  # Check if the element is numeric
                    from .distributions import ConstantDistribution
                    self.matrix.flat[i] = StochasticVariable(
                        ConstantDistribution(element), name=f"_Constant({element})"
                    )
                else:
                    raise ValueError("All elements of StochasticMatrix must be StochasticVariable instances or numeric.")

        self.name = name or "SMat_" + str(id(self))
        self.shape = np.shape(matrix)

    def sample(self, size=1, context=None):
        if context is None:
            context = {}

        samples = np.empty((size,) + self.matrix.shape, dtype=float)
        for idx, var in np.ndenumerate(self.matrix):
            samples[:, idx[0], idx[1]] = var.sample(size=size, context=context)
        return samples[0].tolist() if np.shape(samples)[0] == 1 else samples.tolist()

    def get_all_dependencies(self, visited=None, stack=None):
        dependencies = set()
        for var in self.matrix.flatten():
            dependencies.update(var.get_all_dependencies(visited=visited, stack=stack))
            dependencies.add(var)
        return dependencies

    def transpose(self):
        transposed_matrix = self.matrix.T
        return StochasticMatrix(transposed_matrix, name=f"{self.name}_T")

    @property
    def T(self):
        return self.transpose()

    def matmul(self, other):
        if isinstance(other, StochasticMatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix dimensions do not align for multiplication.")
            result_matrix = np.empty((self.shape[0], other.shape[1]), dtype=object)
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    variables = []
                    for k in range(self.shape[1]):
                        variables.append(apply(operator.mul, self.matrix[i, k], other.matrix[k, j], name=operation_name("*", self.matrix[i, k], other.matrix[k, j])))
                    result_matrix[i, j] = sum(variables)
            return StochasticMatrix(result_matrix, name=operation_name("@", self, other))
        elif isinstance(other, StochasticVector):
            if self.shape[1] != len(other):
                raise ValueError("Matrix and vector dimensions do not align for multiplication.")
            result_vector = np.empty((self.shape[0],), dtype=object)
            for i in range(self.shape[0]):
                # Multiply corresponding elements and sum them
                products = []
                for k in range(self.shape[1]):
                    products.append(self.matrix[i, k] * other.variables[k])
                # Sum the products
                result = reduce(operator.add, products)
                result_vector[i] = result
            return StochasticVector(*result_vector, name=operation_name("@", self, other))
        else:
            raise ValueError("Unsupported type for matrix multiplication.")

    def __matmul__(self, other):
        return self.matmul(other)

    def __str__(self):
        rows = [
            "[{}]".format(", ".join(f"{str(element):>{max(len(str(el)) for row in self.matrix for el in row)}}"
                                    for element in row))
            for row in self.matrix
        ]
        return "\n" + "[" + ",\n ".join(rows) + "]"

    def __getitem__(self, key):
        element = self.matrix[key]
        if isinstance(element, np.ndarray):
            return StochasticMatrix(element, name=f"{self.name}[{key}]")
        else:
            return element

    def __setitem__(self, key, value):
        if isinstance(value, StochasticVariable):
            self.matrix[key] = value
        else:
            raise ValueError("Assigned value must be a StochasticVariable.")



# Core functions


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
    elif isinstance(object, StochasticMatrix):
        for variable in object.matrix.flatten():
            delete(variable)
        del object
    elif isinstance(object, StochasticVector):
        for variable in object.variables:
            delete(variable)
        del object
    return
        

def apply(func, *args, name=None):
    """
    Applies a function to one or more stochastic variables, returning a new stochastic variable.

    Parameters:
        - func: The function to apply.
        - *args: The stochastic variables (or constants) to which the function is applied.
        - name: Optional name for the resulting stochastic variable.

    Returns:
        - StochasticVariable: The result of applying the function.
    """

    from .distributions import ConstantDistribution, TransformedDistribution

    variables = []
    for arg in args:
        if isinstance(arg, StochasticVariable):
            variables.append(arg)
        else:
            variables.append(StochasticVariable(ConstantDistribution(arg), name=f"_Constant({arg})"))

    distribution = TransformedDistribution(func, variables)
    return StochasticVariable(distribution=distribution, name=name)


def probability(condition, *args, size=DEFAULT_STATISTICS_SAMPLE_SIZE, context=None, stochastic=False, name=None):
    
    from .distributions import ProbabilityDistribution
    
    if stochastic:
        return StochasticVariable(ProbabilityDistribution(condition, *args), name=name)

    # Non-stochastic version remains unchanged
    if context is None:
        context = {}

    samples_list = []
    for arg in args:
        if isinstance(arg, StochasticVariable):
            samples = arg.sample(size=size, context=context)
        else:
            samples = np.full(size, arg)
        samples_list.append(samples)

    condition_results = condition(*samples_list)
    condition_results = np.asarray(condition_results, dtype=bool)
    prob = np.mean(condition_results)

    return prob


def set_random_seed(seed):
    """
    Set the random seed for reproducibility in numpy and scipy.
    """
    np.random.seed(seed)
    # Optionally, also configure scipy.stats if applicable
    distributions.rng = np.random.default_rng(seed)



# Helper functions


def operation_name(symbol, term1, term2):
    A = "(" + str(term1) + ")" if any(char in ["+", "-", " "] for char in str(term1)) else str(term1)
    B = "(" + str(term2) + ")" if any(char in ["+", "-", " "] for char in str(term2)) else str(term2)
    return A + " " + symbol + " " + B