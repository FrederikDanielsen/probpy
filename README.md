
# ProbPy

**ProbPy** is a Python library for modeling and manipulating stochastic variables and probability distributions. It provides tools for statistical analysis, transformations, Monte Carlo simulations, and visualizations, making it easier to work with random processes and probabilistic models.

---

## Features

### Stochastic Variables (`StochasticVariable`):
- Define random variables with specified distributions, functions, or constants.
- Support arithmetic operations with other stochastic variables and constants.
- **Statistical methods**:
  - Mean, variance, standard deviation, median, mode, nth moment, confidence intervals.
- **Distribution methods**:
  - Probability density function (PDF), probability mass function (PMF), cumulative distribution function (CDF).
- Empirical methods for variables without explicit PDF/PMF/CDF.
- Circular dependency detection to prevent invalid definitions.

### Stochastic Vectors (`StochasticVector`):
- Define vectors composed of stochastic variables.
- Perform vector operations:
  - Norm (p-norm), dot product, cross product (for 3D vectors).
  - Element-wise arithmetic operations.
- Manage dependencies across all components.

### Distributions:
- **Discrete Distributions**:
  - Bernoulli, Binomial, Geometric, Hypergeometric, Poisson, Negative Binomial, Multinomial, Discrete Uniform.
- **Continuous Distributions**:
  - Normal, Exponential, Gamma, Beta, Chi-Squared, Rayleigh, Cauchy, Dirichlet, Continuous Uniform, Arcsine.
- **Custom Distributions**:
  - Define custom distributions with user-specified probability functions.
  - Support sampling and computing PDF/PMF over specified domains.

### Transformations:
- Apply mathematical functions to stochastic variables, including:
  - Exponential, logarithmic, power, trigonometric, hyperbolic, rounding, and clipping functions.

### Monte Carlo Simulation:
- Perform simulations with stochastic inputs.
- Statistical summarization of simulation results.
- Visualization of simulation outputs.

### Visualization Utilities:
- Plot distributions of stochastic variables.
- Visualize dependency graphs between variables, highlighting circular dependencies.

---

## Installation

Clone the repository and install the package using `pip`:

```bash
git clone https://github.com/FrederikDanielsen/probpy.git
cd probpy
pip install .
```

---

## Getting Started

### Importing the Library

```python
from probpy.core import StochasticVariable, StochasticVector, apply, probability
from probpy.distributions import (
    NormalDistribution, ExponentialDistribution, BernoulliDistribution, CustomDistribution
)
from probpy.transformations import sqrt, log, sin
from probpy.monte_carlo import (
    monte_carlo_simulate, summarize_simulation, plot_simulation
)
from probpy.plots import plot_distribution, plot_dependency_graph
```

### Defining Stochastic Variables

#### Using Standard Distributions

```python
# Normal distribution with mean 0 and standard deviation 1
X = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=1), name='X')

# Exponential distribution with lambda = 1
Y = StochasticVariable(distribution=ExponentialDistribution(lambd=1), name='Y')

# Bernoulli distribution with probability p = 0.5
Z = StochasticVariable(distribution=BernoulliDistribution(p=0.5), name='Z')
```

#### Using Custom Distributions

```python
import numpy as np

# Define a custom PDF function
def custom_pdf(x, a, b):
    return a * np.exp(-b * x)

# Create a custom distribution with parameters a and b
custom_dist = CustomDistribution(func=custom_pdf, a=2, b=1, domain=(0, np.inf))

# Define a stochastic variable with the custom distribution
W = StochasticVariable(distribution=custom_dist, name='W')
```

---

### Performing Arithmetic Operations

```python
# Arithmetic operations between stochastic variables
sum_var = X + Y
diff_var = X - Y
prod_var = X * Y
quot_var = X / Y

# Operations with constants
scaled_var = X * 5
shifted_var = Y + 3
```

---

### Applying Transformations

```python
# Apply mathematical functions to stochastic variables
sqrt_X = sqrt(X)
log_Y = log(Y)
sin_X = sin(X)
```

---

### Working with Stochastic Vectors

```python
# Define stochastic variables
X1 = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=1), name='X1')
X2 = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=1), name='X2')
X3 = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=1), name='X3')

# Create a 3D stochastic vector
vector = StochasticVector(X1, X2, X3, name='Vector')

# Compute the vector norm
vector_norm = vector.norm(p=2)

# Compute the dot product with another vector
Y1 = StochasticVariable(distribution=NormalDistribution(mu=1, sigma=2), name='Y1')
Y2 = StochasticVariable(distribution=NormalDistribution(mu=-1, sigma=2), name='Y2')
Y3 = StochasticVariable(distribution=NormalDistribution(mu=0, sigma=2), name='Y3')
vector2 = StochasticVector(Y1, Y2, Y3, name='Vector2')

dot_product = vector.dot(vector2)

# Compute the cross product (only for 3D vectors)
cross_product = vector.cross(vector2)
```

---

### Monte Carlo Simulation

```python
# Define a model function
def model(x, y, z):
    return x * y + z

# Perform Monte Carlo simulation
variables = [X, Y, Z]
results = monte_carlo_simulate(model, variables, trials=10000, seed=42)

# Summarize the simulation results
summary = summarize_simulation(results)
print("Simulation Summary:")
print(f"Mean: {summary['mean']}")
print(f"Variance: {summary['variance']}")
print(f"Standard Deviation: {summary['std_dev']}")
print(f"Median: {summary['median']}")
print(f"95% Confidence Interval: {summary['confidence_interval']}")
```

---

### Visualizing Results

#### Plotting Distributions

```python
# Plot the distribution of a stochastic variable
plot_distribution(X, num_samples=1000, bins=50, title='Distribution of X')
```

#### Plotting Dependency Graphs

```python
# Visualize dependencies among variables
plot_dependency_graph([sum_var, prod_var, sqrt_X], title='Dependency Graph')
```

---

## Advanced Usage

#### Estimating Probabilities

```python
# Estimate the probability that X > 1 and Y < 2
prob = probability(lambda x, y: (x > 1) & (y < 2), X, Y, size=10000)
print(f"Estimated Probability: {prob}")
```

#### Custom Transformations with `apply()`

```python
# Define a custom function
def custom_function(x, y):
    return x ** 2 + np.sin(y)

# Apply the custom function to stochastic variables
custom_var = apply(custom_function, X, Y, name='CustomVar')
```

---

## Documentation

Detailed documentation is available in the `docs` directory, including API references and tutorials.

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