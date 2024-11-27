
# ProbPy Documentation

## Overview

The ProbPy library provides a comprehensive framework for probabilistic modeling and simulation, enabling the creation and manipulation of stochastic variables and vectors, application of transformations, dependency management, Monte Carlo simulations, and more.

The library is modular, with each component handling specific aspects of probabilistic computations. The main modules include:

- **Core Module**: Defines the core classes `StochasticVariable` and `StochasticVector`, which represent random variables and vectors.
- **Distributions Module**: Provides a variety of probability distributions, both standard and custom, that can be used to define stochastic variables.
- **Transformations Module**: Offers mathematical functions to transform stochastic variables.
- **Monte Carlo Module**: Contains functions for performing Monte Carlo simulations.
- **Plots Module**: Includes functions for visualizing distributions, simulation results, and dependency graphs.
- **Constants Module**: Defines default values used across the library.

---

## Installation

To install ProbPy, clone the repository and install the required dependencies:

```bash
git clone https://github.com/FrederikDanielsen/probpy.git
cd probpy
pip install -r requirements.txt
```

---

## Table of Contents

1. [Core Module](#core-module)
    - [StochasticVariable Class](#stochasticvariable-class)
    - [StochasticVector Class](#stochasticvector-class)
    - [Core Functions](#core-functions)
2. [Distributions Module](#distributions-module)
    - [Base Classes](#base-classes)
    - [Standard Distributions](#standard-distributions)
    - [CustomDistribution Class](#customdistribution-class)
3. [Transformations Module](#transformations-module)
4. [Monte Carlo Module](#monte-carlo-module)
5. [Plots Module](#plots-module)
6. [Constants Module](#constants-module)
7. [Usage Examples](#usage-examples)
8. [License](#license)

---

## Core Module

The core module defines the foundational classes and functions of the library.

### StochasticVariable Class

Represents a stochastic (random) variable.

#### Initialization

```python
StochasticVariable(
    distribution=None,
    dependencies=None,
    func=None,
    name=None,
    distribution_type=None,
    value=None,
)
```

**Parameters**:
- `distribution`: An instance of a distribution (from the Distributions Module).
- `dependencies`: A list of other `StochasticVariable` instances this variable depends on.
- `func`: A callable that generates values based on dependencies.
- `name`: Optional name of the stochastic variable.
- `distribution_type`: `'continuous'`, `'discrete'`, or `'mixed'`.
- `value`: If the variable represents a constant, this is its value.

#### Methods

- `sample(size=1, context=None)`: Generates samples from the stochastic variable.
- `pdf(x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, bandwidth='scott', context=None)`: Computes the probability density function at `x`.
- `pmf(x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, context=None)`: Computes the probability mass function at `x`.
- `cdf(x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, context=None)`: Computes the cumulative distribution function at `x`.
- `mean(size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Calculates the mean of the variable.
- `std(size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Calculates the standard deviation.
- `var(size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Calculates the variance.
- `median(size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Calculates the median.
- `mode(size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Estimates the mode.
- `nth_moment(n, size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Calculates the nth moment.
- `confidence_interval(confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Computes the confidence interval.

#### Arithmetic Operations

Supports overloaded arithmetic operators for combining stochastic variables:

- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Exponentiation: `**`

---

### StochasticVector Class

Represents a stochastic vector composed of multiple stochastic variables.

#### Initialization

```python
StochasticVector(*variables, name=None)
```

**Parameters**:
- `variables`: Instances of `StochasticVariable` to include in the vector.
- `name`: Optional name of the stochastic vector.

#### Methods

- `sample(size=1, context=None)`: Samples from all component variables.
- `norm(p=2)`: Computes the p-norm of the stochastic vector.
- `dot(other)`: Computes the dot product with another stochastic vector.
- `cross(other)`: Computes the cross product with another 3D stochastic vector.

#### Element-wise Operations

Supports element-wise arithmetic operations with other vectors, variables, or scalars:

- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`

---

### Core Functions

#### `apply`

Applies a custom function to one or more stochastic variables.

```python
apply(func, *args, name=None)
```

**Parameters**:
- `func`: A callable to apply.
- `*args`: Stochastic variables or constants to pass to the function.
- `name`: Optional name for the resulting stochastic variable.

#### `probability`

Estimates the probability that a given condition involving stochastic variables is True.

```python
probability(condition, *args, size=DEFAULT_STATISTICS_SAMPLE_SIZE, context=None)
```

**Parameters**:
- `condition`: A callable that returns a boolean array.
- `*args`: Stochastic variables or constants used in the condition.
- `size`: Number of samples to generate.
- `context`: Optional context for sample caching.

---

## Distributions Module

Provides classes for various probability distributions.

### Base Classes

#### `Distribution`

An abstract base class for all distributions.

#### `ParametricDistribution`

A base class for parametric distributions that handles parameter resolution and dependencies.

#### `StandardDistribution`

A class for standard distributions, typically from `scipy.stats`.

### Standard Distributions

Below are some of the standard distributions provided:

#### Discrete Distributions

- `DiscreteUniformDistribution(a, b)`
- `BernoulliDistribution(p)`
- `BinomialDistribution(n, p)`
- `GeometricDistribution(p)`
- `HypergeometricDistribution(M, n, N)`
- `PoissonDistribution(mu)`
- `NegativeBinomialDistribution(n, p)`
- `MultinomialDistribution(n, p)`

#### Continuous Distributions

- `ContinuousUniformDistribution(a, b)`
- `ExponentialDistribution(lambd=1)`
- `NormalDistribution(mu=0, sigma=1)`
- `LogNormalDistribution(s, scale=np.exp(0))`
- `GammaDistribution(shape, scale=1)`
- `ChiSquaredDistribution(df)`
- `RayleighDistribution(scale=1)`
- `BetaDistribution(a, b)`
- `CauchyDistribution(x0=0, gamma=1)`
- `StandardArcsineDistribution()`
- `DirichletDistribution(alpha)`

Each distribution class provides methods for sampling, calculating PDF/PMF, and CDF, depending on whether the distribution is continuous or discrete.

#### CustomDistribution Class

Allows defining a custom probability distribution with user-specified behavior.

#### Initialization

```python
CustomDistribution(func, domain=None, distribution_type='continuous', **params)
```

**Parameters**:
- `func`: A callable representing the PDF (for continuous) or PMF (for discrete).
- `domain`: Tuple specifying the domain of the distribution.
- `distribution_type`: `'continuous'` or `'discrete'`.
- `**params`: Additional parameters required by `func`.


---

## Transformations Module

### Overview
The transformations module provides a suite of mathematical functions that can be applied to stochastic variables. These transformations allow you to compute derived quantities, analyze results, or preprocess data for further analysis.

### Available Transformations

#### Exponential and Logarithmic Functions
- `exp(X)`: Computes the exponential of `X`.
- `expm1(X)`: Computes `exp(X) - 1`.
- `log(X)`: Computes the natural logarithm of `X`.
- `log10(X)`: Computes the base-10 logarithm of `X`.
- `log2(X)`: Computes the base-2 logarithm of `X`.
- `log1p(X)`: Computes the natural logarithm of `1 + X`.

#### Power Functions
- `sqrt(X)`: Computes the square root of `X`.
- `square(X)`: Computes the square of `X`.
- `power(X, y)`: Computes `X` raised to the power of `y`.
- `cbrt(X)`: Computes the cube root of `X`.
- `reciprocal(X)`: Computes the reciprocal of `X`.

#### Trigonometric Functions
- `sin(X)`, `cos(X)`, `tan(X)`: Standard trigonometric functions.
- `arcsin(X)`, `arccos(X)`, `arctan(X)`: Inverse trigonometric functions.
- `arctan2(X, Y)`: Computes the angle between the positive x-axis and the line to the point `(X, Y)`.
- `hypot(X, Y)`: Computes the Euclidean norm.

#### Hyperbolic Functions
- `sinh(X)`, `cosh(X)`, `tanh(X)`: Hyperbolic functions.
- `arcsinh(X)`, `arccosh(X)`, `arctanh(X)`: Inverse hyperbolic functions.

#### Rounding and Clipping Functions
- `round_(X, decimals=0)`: Rounds `X` to the specified number of decimals.
- `floor(X)`: Computes the floor of `X`.
- `ceil(X)`: Computes the ceiling of `X`.
- `trunc(X)`: Truncates `X` to an integer.
- `clip(X, a_min, a_max)`: Clamps the values of `X` within `[a_min, a_max]`.

#### Sign and Comparison Functions
- `abs_(X)`: Computes the absolute value of `X`.
- `sign(X)`: Returns the sign of `X`.
- `min_(*Xs)`: Computes the minimum value among the inputs.
- `max_(*Xs)`: Computes the maximum value among the inputs.

---

## Monte Carlo Module

### Overview
This module enables Monte Carlo simulations for probabilistic modeling. It provides utilities for generating results based on models and summarizing or visualizing the outcomes.

### Functions

#### `monte_carlo_simulate`
Performs Monte Carlo simulation for a given model.

**Parameters**:
- `model`: A callable that defines the simulation model.
- `variables`: List of stochastic variables used as inputs to the model.
- `trials`: Number of Monte Carlo trials (default: `10,000`).
- `seed`: Random seed for reproducibility.

#### `summarize_simulation`
Summarizes simulation results with basic statistics.

**Parameters**:
- `results`: Array of simulation results.
- `confidence_level`: Confidence level for the confidence interval.

#### `plot_simulation`
Plots the distribution of Monte Carlo simulation results.

**Parameters**:
- `results`: Array of simulation results.
- `bins`: Number of bins in the histogram.
- `density`: Whether to normalize the histogram.
- `title`: Title of the plot.

---

## Plots Module

### Overview
Provides utilities for visualizing stochastic variables and their dependencies.

### Functions

#### `plot_distribution`
Plots the distribution of a `StochasticVariable`.

**Parameters**:
- `stochastic_var`: The variable to plot.
- `num_samples`: Number of samples to generate for the plot.
- `bins`: Number of bins for the histogram.
- `density`: Whether to normalize the histogram.
- `title`: Title of the plot.

#### `plot_dependency_graph`
Visualizes the dependency graph of stochastic variables and vectors.

**Parameters**:
- `variables`: List of `StochasticVariable` or `StochasticVector` instances.
- `title`: Title of the graph.

---

## Constants Module

### Overview
Defines default constants used throughout the library.

### Available Constants
- `DEFAULT_STATISTICS_SAMPLE_SIZE`: Default sample size for statistical calculations.
- `DEFAULT_PLOTTING_SAMPLE_SIZE`: Default sample size for plotting.

---

## Usage Examples

### Example 1: Creating and Sampling a Stochastic Variable
```python
from probpy.core import StochasticVariable
from probpy.distributions import NormalDistribution

# Define a stochastic variable
X = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=1))

# Generate samples
samples = X.sample(size=1000)

# Compute statistics
mean = X.mean()
std_dev = X.std()
```

### Example 2: Monte Carlo Simulation
```python
from probpy.monte_carlo import monte_carlo_simulate, summarize_simulation
from probpy.core import StochasticVariable
from probpy.distributions import NormalDistribution

# Define stochastic variables
X = StochasticVariable(distribution=NormalDistribution(mu=5, sigma=2))
Y = StochasticVariable(distribution=NormalDistribution(mu=10, sigma=3))

# Define the model
def model(x_samples, y_samples):
    return x_samples + y_samples

# Perform simulation
results = monte_carlo_simulate(model, [X, Y])

# Summarize results
summary = summarize_simulation(results)
print(summary)
```

### Example 3: Plotting a Dependency Graph
```python
from probpy.plots import plot_dependency_graph
from probpy.core import StochasticVariable
from probpy.distributions import NormalDistribution

# Create dependent variables
X = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=1))
Y = X + 3

# Plot dependency graph
plot_dependency_graph([Y])
```

---

## License

This library is provided under the MIT License. You are free to use, modify, and distribute it as per the terms of the license. [View LICENSE](LICENSE.md)


## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your_feature`
3. Commit your changes: `git commit -am 'Add a new feature'`
4. Push to the branch: `git push origin feature/your_feature`
5. Create a pull request.

---


## Acknowledgments

We appreciate the use of open-source libraries like NumPy, SciPy, Matplotlib, and NetworkX that make ProbPy possible.

---

## Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainer at [danielsen.contact@gmail.com](mailto:danielsen.contact@gmail.com).