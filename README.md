
# ProbPy

**ProbPy** is a Python library for probabilistic modeling, distribution analysis, and stochastic simulations. It provides intuitive and flexible tools for working with stochastic variables, distributions, transformations, and visualization. The library supports various statistical operations and is built with extensibility and ease of use in mind.

---

## Features

- **Stochastic Variables and Vectors**:
  - Model stochastic behavior with `StochasticVariable` and `StochasticVector` classes.
  - Perform operations like arithmetic, statistical analysis, and transformations.

- **Distributions**:
  - Support for discrete and continuous distributions such as:
    - Discrete: Bernoulli, Binomial, Geometric, Poisson, etc.
    - Continuous: Normal, Exponential, Gamma, Beta, etc.
  - Create custom and mixture distributions.

- **Transformations and Functions**:
  - Mathematical transformations (e.g., exponential, logarithm, trigonometric).
  - Operations like norms, dot products, and cross products for vectors.

- **Visualization**:
  - Plot histograms, density functions, and dependency graphs.

- **Goodness of Fit**:
  - Perform tests like Kolmogorov-Smirnov (KS), Anderson-Darling, and Chi-Square.

- **Monte Carlo Simulations**:
  - Run simulations for probabilistic models and summarize results.

---

## Installation

To install the library, clone the repository and use pip:

```bash
git clone https://github.com/FrederikDanielsen/probpy.git
cd probpy
pip install .
```

---

## Getting Started

### Importing the Library

```python
from probpy.core import StochasticVariable, StochasticVector
from probpy.distributions import NormalDistribution, ExponentialDistribution
from probpy.transformations import sqrt, log
from probpy.monte_carlo import monte_carlo_simulate
```

### Example Usage

#### 1. Create and Sample a Distribution

```python
from probpy.distributions import NormalDistribution

# Create a normal distribution with mean=0, std=1
normal_var = NormalDistribution(mu=0, sigma=1)
samples = normal_var.sample(size=1000)
```

#### 2. Perform Arithmetic Operations

```python
from probpy.core import StochasticVariable

# Create variables
a = StochasticVariable(value=5)
b = StochasticVariable(value=10)

# Perform addition
c = a + b
print(c.sample(size=5))  # Output: [15, 15, 15, 15, 15]
```

#### 3. Monte Carlo Simulation

```python
from probpy.monte_carlo import monte_carlo_simulate
from probpy.core import StochasticVariable

# Create variables
a = StochasticVariable(value=5)
b = StochasticVariable(value=10)

# Define a model
def model(x, y):
    return x * y

# Run simulation
results = monte_carlo_simulate(model, [a, b], trials=1000)
print("Mean result:", results.mean())
```

#### 4. Visualization

```python
from probpy.plots import plot_distribution

# Plot a normal variable
plot_distribution(normal_var, num_samples=1000, title="Normal Distribution")
```

---

## Documentation

For detailed documentation and API references, check the [documentation file](./documentation.md).

---

## Testing

Run the tests to verify the library's functionality:

```bash
pytest tests/
```

---

## Contributing

We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature-name"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Acknowledgments

Built with ❤️ using Python and the following libraries:
- NumPy
- SciPy
- Matplotlib
- NetworkX
