# distributions.py

import numpy as np
from scipy.stats import bernoulli, binom, geom, hypergeom, poisson, nbinom, multinomial, randint
from scipy.stats import uniform, expon, norm, gamma, chi2, rayleigh, beta, cauchy, arcsine, dirichlet
from scipy.stats import mode as scipy_mode
from abc import ABC, abstractmethod


class Distribution(ABC):
    """
    Base class for all distributions.
    """

    @abstractmethod
    def sample(self):
        """Generate a random sample from the distribution."""
        pass


class DiscreteDistribution(Distribution):
    """
    Base class for discrete distributions.
    """

    @abstractmethod
    def pmf(self, x):
        """Calculate the probability mass function."""
        pass

    @abstractmethod
    def cdf(self, x):
        """Calculate the cumulative distribution function."""
        pass


class ContinuousDistribution(Distribution):
    """
    Base class for continuous distributions.
    """

    @abstractmethod
    def pdf(self, x):
        """Calculate the probability density function."""
        pass

    @abstractmethod
    def cdf(self, x):
        """Calculate the cumulative distribution function."""
        pass


class StochasticVariable:
    """
    A stochastic variable that wraps a distribution and supports arithmetic operations and statistics.
    """

    def __init__(self, distribution, name=None):
        """
        Initialize a stochastic variable with a given distribution.

        Parameters:
            distribution (Distribution): An instance of a discrete or continuous distribution.
            name (str): Optional name for the variable.
        """
        if not isinstance(distribution, Distribution):
            raise TypeError("distribution must be an instance of Distribution")
        self.__distribution = distribution  # Private distribution
        self.name = name or "Unnamed"  # Optional name for the variable
        self.statistic_sample_size = 1000  # Default sample size for statistics

    def sample(self, size=1):
        """
        Generate one or more random samples from the associated distribution.

        Parameters:
            size (int): Number of samples to generate (default: 1).

        Returns:
            A single sample if size=1, otherwise a NumPy array of samples.
        """
        if size == 1:
            return self.__distribution.sample()
        return np.array(self.__distribution.sample(size=size))

    def _apply_operation(self, other, operation):
        """
        Apply an operation (+, -, *, /, **, %) between a stochastic variable and another variable (stochastic or scalar).

        Parameters:
            other (StochasticVariable or scalar): The other variable in the operation.
            operation (callable): A function implementing the operation.

        Returns:
            StochasticVariable: A new stochastic variable representing the result.
        """
        if isinstance(other, (int, float)):  # Handle scalar operations
            class ScalarCompositeDistribution(Distribution):
                def __init__(self, dist, scalar, operation):
                    self.dist = dist
                    self.scalar = scalar
                    self.operation = operation

                def sample(self, size=1):
                    if size == 1:
                        return self.operation(self.dist.sample(), self.scalar)
                    return self.operation(self.dist.sample(size=size), self.scalar)

            return StochasticVariable(
                ScalarCompositeDistribution(self.__distribution, other, operation),
                name=f"{self.name} {operation.__name__} {other}",
            )

        elif isinstance(other, StochasticVariable):  # Handle stochastic variable operations
            class CompositeDistribution(Distribution):
                def __init__(self, dist1, dist2, operation):
                    self.dist1 = dist1
                    self.dist2 = dist2
                    self.operation = operation

                def sample(self, size=1):
                    if size == 1:
                        return self.operation(self.dist1.sample(), self.dist2.sample())
                    return self.operation(
                        self.dist1.sample(size=size),
                        self.dist2.sample(size=size),
                    )

            # Map the operation to its symbol
            operation_symbols = {
                np.add: "+",
                np.subtract: "-",
                np.multiply: "*",
                np.divide: "/",
                np.power: "**",
                np.mod: "%",
            }
            symbol = operation_symbols.get(operation, "?")  # Default to "?" if not found

            return StochasticVariable(
                CompositeDistribution(self.__distribution, other.__distribution, operation),
                name=f"{self.name} {symbol} {other.name}",
            )

        else:
            raise TypeError("Can only operate with StochasticVariable or scalar.")

    def confidence_interval(self, confidence_level=0.95, size=None):
        if size is None:
            size = self.statistic_sample_size

        if not (0 < confidence_level <= 1):
            raise ValueError("Confidence level must be in the range (0, 1].")

        samples = self.sample(size=size)
        lower_quantile = (1 - confidence_level) / 2
        upper_quantile = 1 - lower_quantile
        lower_bound = np.quantile(samples, lower_quantile)
        upper_bound = np.quantile(samples, upper_quantile)

        return lower_bound, upper_bound

    def mean(self, size=None):
        if size is None:
            size = self.statistic_sample_size
        samples = self.sample(size=size)
        return np.mean(samples)

    def std(self, size=None):
        if size is None:
            size = self.statistic_sample_size
        samples = self.sample(size=size)
        return np.std(samples)

    def median(self, size=None):
        if size is None:
            size = self.statistic_sample_size
        samples = self.sample(size=size)
        return np.median(samples)

    def mode(self, size=None):
        if size is None:
            size = self.statistic_sample_size
        samples = self.sample(size=size)
        mode_result = scipy_mode(samples, axis=None)
        return mode_result.mode[0]

    def moment(self, n, size=None):
        if size is None:
            size = self.statistic_sample_size
        samples = self.sample(size=size)
        return np.mean(samples ** n)

    def __add__(self, other):
        return self._apply_operation(other, np.add)

    def __sub__(self, other):
        return self._apply_operation(other, np.subtract)

    def __mul__(self, other):
        return self._apply_operation(other, np.multiply)

    def __truediv__(self, other):
        return self._apply_operation(other, np.divide)

    def __pow__(self, other):
        return self._apply_operation(other, np.power)

    def __mod__(self, other):
        return self._apply_operation(other, np.mod)

    def __repr__(self):
        return f"StochasticVariable(name={self.name}, distribution={self.__distribution.__class__.__name__})"




# Discrete Distributions

class DiscreteUniformDistribution(DiscreteDistribution):
    """
    Discrete Uniform Distribution: Models a uniform distribution over integers [a, b].
    """
    def __init__(self, a, b):
        # For scipy.stats.randint, low=a and high=b+1 to include b.
        self.parameters = [a, b + 1]  # Adjust for inclusive range
        self.distribution = randint

    def _get_parameters(self):
        # Resolve parameters, evaluating StochasticVariable if necessary
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        # Check if any parameter is a StochasticVariable
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self.parameters).rvs() for _ in range(size)]
        else:
            return self.distribution(*self.parameters).rvs(size=size)

    def pmf(self, x):
        # Probability mass function
        return self.distribution(*self.parameters).pmf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self.parameters).cdf(x)


class BernoulliDistribution(DiscreteDistribution):
    """
    Bernoulli Distribution: Models a single trial with a success probability p.
    """
    def __init__(self, p):
        self.parameters = [p]
        self.distribution = bernoulli

    def _get_parameters(self):
        # Resolve parameters, evaluating StochasticVariable if necessary
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        # Check if any parameter is a StochasticVariable
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(self._get_parameters()[0]).rvs() for _ in range(size)]
        else:
            return self.distribution(self._get_parameters()[0]).rvs(size=size)

    def pmf(self, x):
        # Probability mass function
        return self.distribution(self._get_parameters()[0]).pmf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(self._get_parameters()[0]).cdf(x)


class BinomialDistribution(DiscreteDistribution):
    """
    Binomial Distribution: Models the number of successes in n independent trials
    with a success probability p.
    """
    def __init__(self, n, p):
        self.parameters = [n, p]
        self.distribution = binom

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)

    def pmf(self, x):
        # Probability mass function
        return self.distribution(*self._get_parameters()).pmf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self._get_parameters()).cdf(x)


class GeometricDistribution(DiscreteDistribution):
    """
    Geometric Distribution: Models the number of trials until the first success
    in a sequence of independent Bernoulli trials with success probability p.
    """
    def __init__(self, p):
        self.parameters = [p]
        self.distribution = geom

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(self._get_parameters()[0]).rvs() for _ in range(size)]
        else:
            return self.distribution(self._get_parameters()[0]).rvs(size=size)

    def pmf(self, x):
        # Probability mass function
        return self.distribution(self._get_parameters()[0]).pmf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(self._get_parameters()[0]).cdf(x)


class HypergeometricDistribution(DiscreteDistribution):
    """
    Hypergeometric Distribution: Models the number of successes in a sample of size n
    from a population of size N with k successes, without replacement.
    """
    def __init__(self, N, K, n):
        self.parameters = [N, K, n]
        self.distribution = hypergeom

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)

    def pmf(self, x):
        # Probability mass function
        return self.distribution(*self._get_parameters()).pmf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self._get_parameters()).cdf(x)


class PoissonDistribution(DiscreteDistribution):
    """
    Poisson Distribution: Models the number of events occurring in a fixed interval of time
    or space, given a known constant mean rate (lambda_).
    """
    def __init__(self, lambda_):
        self.parameters = [lambda_]
        self.distribution = poisson

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(self._get_parameters()[0]).rvs() for _ in range(size)]
        else:
            return self.distribution(self._get_parameters()[0]).rvs(size=size)

    def pmf(self, x):
        # Probability mass function
        return self.distribution(self._get_parameters()[0]).pmf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(self._get_parameters()[0]).cdf(x)


class NegativeBinomialDistribution(DiscreteDistribution):
    """
    Negative Binomial Distribution: Models the number of trials needed to achieve
    r successes in a sequence of independent Bernoulli trials with success probability p.
    """
    def __init__(self, r, p):
        self.parameters = [r, p]
        self.distribution = nbinom

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)

    def pmf(self, x):
        # Probability mass function
        return self.distribution(*self._get_parameters()).pmf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self._get_parameters()).cdf(x)


class MultinomialDistribution(DiscreteDistribution):
    """
    Multinomial Distribution: Models the counts of outcomes in n independent trials,
    where each trial has k possible outcomes with specified probabilities (pvals).
    """
    def __init__(self, n, pvals):
        self.parameters = [n, pvals]
        self.distribution = multinomial

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)





# Continuous Distributions


class ContinuousUniformDistribution(ContinuousDistribution):
    """
    Continuous Uniform Distribution: Models a uniform distribution over the interval [a, b].
    """
    def __init__(self, a, b):
        # For scipy.stats.uniform, loc=a and scale=(b-a).
        self.parameters = [a, b - a]
        self.distribution = uniform

    def _get_parameters(self):
        # Resolve parameters, evaluating StochasticVariable if necessary
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        # Check if any parameter is a StochasticVariable
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self.parameters).rvs() for _ in range(size)]
        else:
            return self.distribution(*self.parameters).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(*self.parameters).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self.parameters).cdf(x)


class ExponentialDistribution(ContinuousDistribution):
    """
    Exponential Distribution: Models the time between events in a Poisson process
    with rate parameter λ (lambda).
    """
    def __init__(self, lambda_):
        self.parameters = [lambda_]  # λ is the rate parameter
        self.distribution = expon

    def _get_parameters(self):
        # Resolve λ (rate parameter) and convert to scale = 1/λ
        lambda_ = [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters][0]
        return 1 / lambda_

    def _has_stochastic_parameter(self):
        # Check if λ is a StochasticVariable
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        scale = self._get_parameters()
        if self._has_stochastic_parameter():
            return [self.distribution(scale=scale).rvs() for _ in range(size)]
        else:
            return self.distribution(scale=scale).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        scale = self._get_parameters()
        return self.distribution(scale=scale).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        scale = self._get_parameters()
        return self.distribution(scale=scale).cdf(x)


class NormalDistribution(ContinuousDistribution):
    """
    Normal Distribution: Models the normal (Gaussian) distribution with mean (mu) and standard deviation (sigma).
    """
    def __init__(self, mu, sigma):
        self.parameters = [mu, sigma]
        self.distribution = norm

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(*self._get_parameters()).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self._get_parameters()).cdf(x)


class GammaDistribution(ContinuousDistribution):
    """
    Gamma Distribution: Models a gamma distribution with shape parameter α (alpha)
    and rate parameter λ (lambda).
    """
    def __init__(self, alpha, lambda_):
        self.parameters = [alpha, lambda_]  # α is the shape, λ is the rate
        self.distribution = gamma

    def _get_parameters(self):
        # Resolve α and λ, convert λ to scale = 1 / λ
        alpha, lambda_ = [
            x.sample() if isinstance(x, StochasticVariable) else x
            for x in self.parameters
        ]
        scale = 1 / lambda_  # Convert λ (rate) to scale
        return alpha, scale

    def _has_stochastic_parameter(self):
        # Check if any parameter is a StochasticVariable
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        alpha, scale = self._get_parameters()
        if self._has_stochastic_parameter():
            return [self.distribution(alpha, scale=scale).rvs() for _ in range(size)]
        else:
            return self.distribution(alpha, scale=scale).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        alpha, scale = self._get_parameters()
        return self.distribution(alpha, scale=scale).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        alpha, scale = self._get_parameters()
        return self.distribution(alpha, scale=scale).cdf(x)


class ChiSquaredDistribution(ContinuousDistribution):
    """
    Chi-Squared Distribution: Models a chi-squared distribution with degrees of freedom (df).
    """
    def __init__(self, df):
        self.parameters = [df]
        self.distribution = chi2

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(self._get_parameters()[0]).rvs() for _ in range(size)]
        else:
            return self.distribution(self._get_parameters()[0]).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(self._get_parameters()[0]).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(self._get_parameters()[0]).cdf(x)


class RayleighDistribution(ContinuousDistribution):
    """
    Rayleigh Distribution: Models a Rayleigh distribution with scale parameter (sigma).
    """
    def __init__(self, sigma):
        self.parameters = [sigma]
        self.distribution = rayleigh

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(self._get_parameters()[0]).rvs() for _ in range(size)]
        else:
            return self.distribution(self._get_parameters()[0]).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(self._get_parameters()[0]).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(self._get_parameters()[0]).cdf(x)


class BetaDistribution(ContinuousDistribution):
    """
    Beta Distribution: Models a Beta distribution with shape parameters (alpha, beta).
    """
    def __init__(self, alpha, beta):
        self.parameters = [alpha, beta]
        self.distribution = beta

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(*self._get_parameters()).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self._get_parameters()).cdf(x)


class CauchyDistribution(ContinuousDistribution):
    """
    Cauchy Distribution: Models a Cauchy distribution with location (x_0) and scale (gamma).
    """
    def __init__(self, x_0, gamma):
        self.parameters = [x_0, gamma]
        self.distribution = cauchy

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(*self._get_parameters()).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self._get_parameters()).cdf(x)


class ArcsineDistribution(ContinuousDistribution):
    """
    Arcsine Distribution: Models an arcsine distribution over [a, b].
    """
    def __init__(self, a, b):
        self.parameters = [a, b]
        self.distribution = arcsine

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(*self._get_parameters()).rvs() for _ in range(size)]
        else:
            return self.distribution(*self._get_parameters()).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(*self._get_parameters()).pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution(*self._get_parameters()).cdf(x)


class StandardArcsineDistribution(ContinuousDistribution):
    """
    Standard Arcsine Distribution: Models the standard arcsine distribution over (0, 1).
    """
    def __init__(self):
        self.distribution = arcsine  # scipy.stats.arcsine defaults to loc=0, scale=1 for (0, 1)

    def sample(self, size=1):
        # Sample from the standard arcsine distribution
        return self.distribution().rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution().pdf(x)

    def cdf(self, x):
        # Cumulative distribution function
        return self.distribution().cdf(x)


class DirichletDistribution(ContinuousDistribution):
    """
    Dirichlet Distribution: Models a Dirichlet distribution with a vector of concentration parameters (alpha).
    """
    def __init__(self, alpha):
        self.parameters = [alpha]
        self.distribution = dirichlet

    def _get_parameters(self):
        return [x.sample() if isinstance(x, StochasticVariable) else x for x in self.parameters]

    def _has_stochastic_parameter(self):
        return any(isinstance(x, StochasticVariable) for x in self.parameters)

    def sample(self, size=1):
        # Sample from the distribution
        if self._has_stochastic_parameter():
            return [self.distribution(self._get_parameters()[0]).rvs() for _ in range(size)]
        else:
            return self.distribution(self._get_parameters()[0]).rvs(size=size)

    def pdf(self, x):
        # Probability density function
        return self.distribution(self._get_parameters()[0]).pdf(x)




