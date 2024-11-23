
# probpy Library Documentation

## Overview
`probpy` is a Python library for working with probability distributions and stochastic variables. It simplifies sampling, performing arithmetic operations, and calculating probabilities for stochastic variables, leveraging the power of `scipy.stats` under the hood.

The core components of this library are:
- The **`StochasticVariable`** class, which represents a random variable associated with a probability distribution.
- The **`probability`** function, which computes the empirical probability of events involving stochastic variables.

---

## StochasticVariable Class

### Overview
The **`StochasticVariable`** class encapsulates a random variable defined by a probability distribution. It allows:
1. **Sampling**: Draw random samples from the associated distribution.
2. **Arithmetic Operations**: Combine stochastic variables or perform operations with scalars (`+`, `-`, `*`, `/`, etc.).
3. **Statistics**: Compute empirical statistics like mean, standard deviation, confidence intervals, and moments.
4. **Dynamic Parameters**: Use other `StochasticVariable` objects as parameters for the distribution.
5. **Integration with the `probability` Function**: Evaluate the likelihood of events involving one or more stochastic variables.

---

### Creating a Stochastic Variable

#### Description
Creates a stochastic variable from a specified probability distribution.

#### Prototype
```python
StochasticVariable(distribution, name=None)
```

#### Parameters
- `distribution`: An instance of a supported probability distribution (e.g., `NormalDistribution`, `BinomialDistribution`).
- `name (str)`: An optional name for the stochastic variable.

#### Example
```python
from probpy.distributions import StochasticVariable, NormalDistribution

# Create a stochastic variable from a normal distribution
X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name="X")
```

---

### Sampling

#### Description
Generates random samples from the underlying probability distribution.

#### Prototype
```python
StochasticVariable.sample(size=1)
```

#### Parameters
- `size (int)`: Number of samples to generate. Default is `1`.

#### Returns
- A single sample if `size=1`; otherwise, a NumPy array of samples.

#### Example
```python
# Sample a single value
single_sample = X.sample()
print("Single sample:", single_sample)

# Sample multiple values
samples = X.sample(size=5)
print("Samples:", samples)
```

---

### Arithmetic Operations

#### Description
Supports arithmetic operations (`+`, `-`, `*`, `/`, `**`, `%`) between stochastic variables and scalars.

#### Example
```python
# Define stochastic variables
Y = StochasticVariable(NormalDistribution(mu=1, sigma=2), name="Y")

# Arithmetic operations
Z = X + Y  # Sum of two stochastic variables
W = 2 * X - Y  # Linear combination of variables
print("Z:", Z)
print("W:", W)

# Sampling from the resulting variable
z_samples = Z.sample(size=5)
print("Samples from Z:", z_samples)
```

---

### Statistical Methods

#### Description
The `StochasticVariable` class provides methods to compute empirical statistics based on samples.

#### Prototypes and Descriptions
```python
StochasticVariable.mean(size=None)
```
- Computes the empirical mean of the samples.
- `size`: Number of samples to use. Default is the variable's `statistic_sample_size`.

```python
StochasticVariable.std(size=None)
```
- Computes the empirical standard deviation of the samples.

```python
StochasticVariable.median(size=None)
```
- Computes the empirical median of the samples.

```python
StochasticVariable.mode(size=None)
```
- Computes the most common value (mode) of the samples.

```python
StochasticVariable.moment(n, size=None)
```
- Computes the nth moment of the samples.

```python
StochasticVariable.confidence_interval(confidence_level=0.95, size=None)
```
- Computes the confidence interval for the given confidence level.

#### Examples
```python
# Mean
mean = X.mean(size=1000)
print("Mean of X:", mean)

# Standard Deviation
std = X.std(size=1000)
print("Standard Deviation of X:", std)

# Confidence Interval
ci = X.confidence_interval(confidence_level=0.95, size=1000)
print("95% Confidence Interval:", ci)
```

---

### Probability

#### Description
Calculates the empirical probability of an event involving one or more stochastic variables.

#### Prototype
```python
probability(*stochastic_variables, condition, size=None)
```

#### Parameters
- `*stochastic_variables`: One or more `StochasticVariable` objects.
- `condition (callable)`: A function that evaluates a condition for the samples.
- `size (int)`: Number of samples to use. Default is the variable's `statistic_sample_size`.

#### Returns
- `float`: The empirical probability of the event.

#### Example
```python
prob = P(X, Y, condition=lambda x, y: x + y > 2, size=10000)
print("P(X + Y > 2):", prob)
```

---

## Supported Distributions

### Discrete Distributions
- **`DiscreteUniformDistribution(a, b)`**: Uniform distribution over integers `[a, b]`.
- **`BernoulliDistribution(p)`**: Single trial with success probability `p`.
- **`BinomialDistribution(n, p)`**: Number of successes in `n` trials with success probability `p`.
- **`GeometricDistribution(p)`**: Number of trials until the first success.
- **`HypergeometricDistribution(N, K, n)`**: Number of successes in a sample of size `n` from a population of size `N` with `K` successes.
- **`PoissonDistribution(lambda_)`**: Number of events occurring in a fixed interval.
- **`NegativeBinomialDistribution(r, p)`**: Trials needed to achieve `r` successes.
- **`MultinomialDistribution(n, pvals)`**: Counts of outcomes in `n` trials with `k` outcomes.

---

### Continuous Distributions
- **`ContinuousUniformDistribution(a, b)`**: Uniform distribution over `[a, b]`.
- **`ExponentialDistribution(lambda_)`**: Time between events in a Poisson process.
- **`NormalDistribution(mu, sigma)`**: Gaussian distribution with mean `mu` and standard deviation `sigma`.
- **`GammaDistribution(alpha, lambda_)`**: Gamma distribution with shape `alpha` and rate `lambda_`.
- **`ChiSquaredDistribution(df)`**: Chi-squared distribution with degrees of freedom `df`.
- **`RayleighDistribution(sigma)`**: Rayleigh distribution with scale `sigma`.
- **`BetaDistribution(alpha, beta_)`**: Beta distribution with shape parameters `alpha` and `beta_`.
- **`CauchyDistribution(x_0, gamma)`**: Cauchy distribution with location `x_0` and scale `gamma`.
- **`DirichletDistribution(alpha)`**: Dirichlet distribution with concentration parameters `alpha`.

---

## Examples

### Example 1: Sampling and Statistics
```python
from probpy.distributions import StochasticVariable, NormalDistribution

X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name="X")
print("Mean:", X.mean(size=1000))
print("95% Confidence Interval:", X.confidence_interval(0.95, size=1000))
```

### Example 2: Arithmetic Operations
```python
from probpy.distributions import StochasticVariable, BinomialDistribution

X = StochasticVariable(BinomialDistribution(10, 0.5), name="X")
Y = StochasticVariable(BinomialDistribution(5, 0.7), name="Y")
Z = X + 2 * Y
print("Samples from Z:", Z.sample(size=1000))
```

### Example 3: Conditional Probability
```python
from probpy.distributions import StochasticVariable, NormalDistribution
from probpy import probability as P

X = StochasticVariable(NormalDistribution(0, 1), name="X")
Y = StochasticVariable(NormalDistribution(1, 2), name="Y")
prob = P(X, Y, condition=lambda x, y: x + y > 3)
print(f"P(X + Y > 3): {prob}")
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
