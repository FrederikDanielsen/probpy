
# ProbPy Documentation

Welcome to **ProbPy**, a Python library for probabilistic modeling, stochastic variables, and Monte Carlo simulations. ProbPy provides an intuitive interface for creating and manipulating stochastic variables, performing statistical analyses, and visualizing probabilistic models.

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Creating Stochastic Variables](#creating-stochastic-variables)
  - [Performing Operations](#performing-operations)
  - [Sampling and Statistical Methods](#sampling-and-statistical-methods)
- [Core Modules](#core-modules)
  - [Core Classes](#core-classes)
  - [Distributions](#distributions)
  - [Transformations](#transformations)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
  - [Visualization Utilities](#visualization-utilities)
- [Examples](#examples)
  - [Basic Arithmetic Operations](#basic-arithmetic-operations)
  - [Monte Carlo Simulation Example](#monte-carlo-simulation-example)
  - [Plotting Distributions](#plotting-distributions)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

ProbPy is designed to simplify probabilistic computations and simulations. It allows you to define stochastic variables with specified distributions, perform arithmetic and functional transformations, and conduct Monte Carlo simulations with ease. The library supports a wide range of probability distributions, both discrete and continuous, and provides tools for statistical analysis and visualization.

---

## Installation

To install ProbPy, clone the repository and install the required dependencies:

```bash
git clone https://github.com/FrederikDanielsen/probpy.git
cd probpy
pip install -r requirements.txt
```

---

## Getting Started

### Creating Stochastic Variables

You can create stochastic variables using predefined distributions or custom functions.

```python
from probpy.core import StochasticVariable
from probpy.distributions import NormalDistribution, ExponentialDistribution

# Create a normal stochastic variable
X = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=1), name="X")

# Create an exponential stochastic variable
Y = StochasticVariable(distribution=ExponentialDistribution(lambd=1), name="Y")
```

### Performing Operations

ProbPy supports arithmetic operations between stochastic variables and constants.

```python
# Addition
Z = X + Y

# Multiplication
W = X * 5

# Custom function application
from probpy.core import apply

def custom_function(x, y):
    return x ** 2 + y ** 2

R = apply(custom_function, X, Y, name="R")
```

### Sampling and Statistical Methods

You can generate samples and compute statistical properties.

```python
# Generate samples
samples = Z.sample(size=1000)

# Compute mean and standard deviation
mean_Z = Z.mean()
std_Z = Z.std()

# Compute empirical PDF
pdf_values = Z.pdf(samples)

# Confidence interval
ci_lower, ci_upper = Z.confidence_interval(confidence_level=0.95)
```

---

## Core Modules

### Core Classes

#### `StochasticVariable`

Represents a random variable with support for:

- Sampling from distributions.
- Arithmetic operations.
- Statistical methods (`mean()`, `std()`, `var()`, `median()`, `confidence_interval()`, etc.).
- Conditioning on other variables.
- Plotting distributions.

#### `StochasticVector`

Represents a vector of stochastic variables with support for:

- Vector operations (`norm()`, `dot()`, `cross()`).
- Element-wise arithmetic operations.

### Distributions

ProbPy provides a variety of probability distributions.

#### Continuous Distributions

- `NormalDistribution(mu, sigma)`
- `ExponentialDistribution(lambd)`
- `GammaDistribution(shape, scale)`
- `BetaDistribution(alpha, beta)`
- `ChiSquaredDistribution(df)`
- `RayleighDistribution(scale)`
- `CauchyDistribution(loc, scale)`
- `ContinuousUniformDistribution(a, b)`
- `StandardArcsineDistribution()`

#### Discrete Distributions

- `BernoulliDistribution(p)`
- `BinomialDistribution(n, p)`
- `GeometricDistribution(p)`
- `HypergeometricDistribution(M, n, N)`
- `PoissonDistribution(mu)`
- `NegativeBinomialDistribution(n, p)`
- `MultinomialDistribution(n, p)`
- `DiscreteUniformDistribution(a, b)`

#### Custom Distributions

- `CustomDistribution(func, domain, distribution_type)`

#### Mixture Distribution

- `MixtureDistribution(components, weights)`

### Transformations

Functional transformations are available for stochastic variables.

```python
from probpy.transformations import exp, log, sqrt, sin

# Apply transformations
exp_X = exp(X)
log_Y = log(Y)
sqrt_Z = sqrt(Z)
sin_W = sin(W)
```

### Monte Carlo Simulation

Perform simulations and analyze results.

```python
from probpy.monte_carlo import monte_carlo_simulate, summarize_simulation

def model(x, y):
    return x * y + x

variables = [X, Y]
results = monte_carlo_simulate(model, variables, trials=10000)
summary = summarize_simulation(results)
```

### Visualization Utilities

#### Plotting Distributions

```python
from probpy.plots import plot_distribution

plot_distribution(X, bins=50, density=True, title="Distribution of X")
```

#### Plotting Dependency Graphs

```python
from probpy.plots import plot_dependency_graph

plot_dependency_graph([Z], title="Dependency Graph of Z")
```

---

## Examples

### Basic Arithmetic Operations

```python
from probpy.core import StochasticVariable
from probpy.distributions import NormalDistribution

# Define stochastic variables
A = StochasticVariable(distribution=NormalDistribution(mu=5, sigma=2), name="A")
B = StochasticVariable(distribution=NormalDistribution(mu=3, sigma=1), name="B")

# Perform operations
C = A + B
D = A * B
E = A / B

# Sample and compute statistics
samples_C = C.sample(size=1000)
mean_C = C.mean()
std_C = C.std()
```

### Monte Carlo Simulation Example

```python
from probpy.core import StochasticVariable
from probpy.distributions import NormalDistribution
from probpy.monte_carlo import monte_carlo_simulate, summarize_simulation

# Define stochastic inputs
demand = StochasticVariable(distribution=NormalDistribution(mu=1000, sigma=100), name="Demand")
price = StochasticVariable(distribution=NormalDistribution(mu=10, sigma=1), name="Price")
cost = StochasticVariable(distribution=NormalDistribution(mu=7, sigma=0.5), name="Cost")

# Define the profit model
def profit(demand, price, cost):
    return demand * (price - cost)

# Perform simulation
variables = [demand, price, cost]
results = monte_carlo_simulate(profit, variables, trials=10000)

# Summarize results
summary = summarize_simulation(results)
print("Expected Profit:", summary["mean"])
print("Profit Variance:", summary["variance"])
```

### Plotting Distributions

```python
from probpy.core import StochasticVariable
from probpy.distributions import ExponentialDistribution
from probpy.plots import plot_distribution

# Define a stochastic variable
lifetime = StochasticVariable(distribution=ExponentialDistribution(lambd=0.1), name="Lifetime")

# Plot its distribution
plot_distribution(lifetime, bins=50, density=True, title="Lifetime Distribution")
```

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your_feature`
3. Commit your changes: `git commit -am 'Add a new feature'`
4. Push to the branch: `git push origin feature/your_feature`
5. Create a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We appreciate the use of open-source libraries like NumPy, SciPy, Matplotlib, and NetworkX that make ProbPy possible.

---

## Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainer at [danielsen.contact@gmail.com](mailto:danielsen.contact@gmail.com).


