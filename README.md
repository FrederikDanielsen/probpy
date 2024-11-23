
# Probpy - Probability Python Library

Probpy is a Python library designed to handle probability-related problems. It provides a convenient interface for defining and manipulating stochastic variables, sampling from probability distributions, performing arithmetic operations on random variables, and calculating empirical probabilities.

## Features

- **StochasticVariable Class**: 
  - Encapsulates a probability distribution as a random variable.
  - Supports sampling, arithmetic operations, and statistical computations.
  - Enables dynamic parameters with stochastic dependencies.

- **Probability Function**: 
  - Calculate the empirical probability of events involving one or more stochastic variables.
  - Supports complex conditions and multiple variables.

- **Extensive Distribution Support**:
  - Discrete distributions like Bernoulli, Binomial, Poisson, and more.
  - Continuous distributions like Normal, Exponential, Gamma, and others.

- **Statistical Methods**:
  - Compute mean, standard deviation, median, mode, nth moment, and confidence intervals.

- **Arithmetic Operations**:
  - Combine stochastic variables or scalars with operators (`+`, `-`, `*`, `/`, etc.).

- **Dynamic Dependencies**:
  - Create stochastic variables whose distributions depend on other stochastic variables.

## Installation

Install Probpy using pip:

```bash
pip install probpy
```

## Quick Start

### Define a Stochastic Variable
```python
from probpy.distributions import StochasticVariable, NormalDistribution

# Create a stochastic variable with a normal distribution
X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name="X")
```

### Sample from a Stochastic Variable
```python
# Generate samples
single_sample = X.sample()
multiple_samples = X.sample(size=5)

print("Single sample:", single_sample)
print("Multiple samples:", multiple_samples)
```

### Perform Arithmetic Operations
```python
from probpy.distributions import StochasticVariable, BinomialDistribution

# Define two stochastic variables
Y = StochasticVariable(BinomialDistribution(10, 0.5), name="Y")

# Combine variables
Z = X + Y * 2

# Sample from the resulting variable
z_samples = Z.sample(size=10)
print("Samples from Z:", z_samples)
```

### Compute Statistics
```python
# Compute mean and confidence interval
mean = X.mean(size=1000)
ci = X.confidence_interval(confidence_level=0.95, size=1000)

print(f"Mean: {mean}")
print(f"95% Confidence Interval: {ci}")
```

### Conditional Probability
```python
from probpy import probability as P

# Compute the probability of an event
prob = P(X, Y, condition=lambda x, y: x + y > 3, size=10000)
print(f"P(X + Y > 3): {prob}")
```

## Supported Distributions

### Discrete Distributions
- **DiscreteUniformDistribution**: Uniform distribution over integers `[a, b]`.
- **BernoulliDistribution**: Single trial with success probability `p`.
- **BinomialDistribution**: Number of successes in `n` trials with success probability `p`.
- **GeometricDistribution**: Number of trials until the first success.
- **PoissonDistribution**: Number of events occurring in a fixed interval.
- **NegativeBinomialDistribution**: Trials needed to achieve `r` successes.
- **MultinomialDistribution**: Counts of outcomes in `n` trials with `k` outcomes.

### Continuous Distributions
- **ContinuousUniformDistribution**: Uniform distribution over `[a, b]`.
- **NormalDistribution**: Gaussian distribution with mean `mu` and standard deviation `sigma`.
- **ExponentialDistribution**: Time between events in a Poisson process.
- **GammaDistribution**: Gamma distribution with shape `alpha` and rate `lambda_`.
- **BetaDistribution**: Beta distribution with shape parameters `alpha` and `beta_`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, create a feature branch, and submit a pull request.

---

## Contact

For questions, suggestions, or support, feel free to reach out to me at danielsen.contact@gmail.com.
