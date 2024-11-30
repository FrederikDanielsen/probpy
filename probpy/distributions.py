# distributions.py

# IMPORTS
import numpy as np
from scipy.stats import norm, lognorm, uniform, expon, poisson, randint, bernoulli, binom, geom, hypergeom, nbinom, multinomial, gamma, chi2, rayleigh, beta, cauchy, arcsine, dirichlet, rv_discrete, rv_continuous, gaussian_kde
from .core import StochasticVariable
from .constants import DEFAULT_STATISTICS_SAMPLE_SIZE
from abc import ABC, abstractmethod
from scipy.stats import t
from scipy.integrate import quad


# Base abstract class for all distributions
class Distribution(ABC):
    def __init__(self, distribution_type=None):
        self.distribution_type = distribution_type

    @abstractmethod
    def sample(self, size=1, context=None):
        pass

    def get_dependencies(self):
        return set()

    def pdf(self, x, context=None):
        if self.distribution_type in ['continuous', 'mixed']:
            raise NotImplementedError("PDF not implemented for this distribution.")
        else:
            raise ValueError("PDF is not defined for discrete distributions.")

    def pmf(self, x, context=None):
        if self.distribution_type in ['discrete', 'mixed']:
            raise NotImplementedError("PMF not implemented for this distribution.")
        else:
            raise ValueError("PMF is not defined for continuous distributions.")

    def cdf(self, x, context=None):
        raise NotImplementedError("CDF not implemented for this distribution.")

    def empirical_pdf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE, bandwidth='scott'):
        if self.distribution_type in ['continuous', 'mixed']:
            samples = self.sample(size=size)
            kde = gaussian_kde(samples, bw_method=bandwidth)
            return kde.evaluate(x)
        else:
            raise ValueError("Empirical PDF is not defined for discrete distributions.")

    def empirical_pmf(self, x, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        if self.distribution_type in ['discrete', 'mixed']:
            samples = self.sample(size=size)
            counts = np.array([np.sum(samples == xi) for xi in np.atleast_1d(x)])
            return counts / size
        else:
            raise ValueError("Empirical PMF is not defined for continuous distributions.")

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

    #def confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
    #    samples = self.sample(size=size)
    #    mean = np.mean(samples)
    #    sem = np.std(samples, ddof=1) / np.sqrt(size)
    #    h = sem * t.ppf((1 + confidence_level) / 2., size - 1)
    #    return mean - h, mean + h
    
    def mean_confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)
        mean = np.mean(samples)
        sem = np.std(samples, ddof=1) / np.sqrt(size)
        h = sem * t.ppf((1 + confidence_level) / 2., size - 1)
        if self.distribution_type == "discrete":
            return np.floor(mean-h), np.ceil(mean+h)
        return mean - h, mean + h
    
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
        return lower_bound, upper_bound

    def confidence_interval(self, confidence_level=0.95, size=DEFAULT_STATISTICS_SAMPLE_SIZE):
        samples = self.sample(size=size)

        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)  # e.g., 2.5% for 95% confidence
        upper_percentile = 100 * (1 - alpha / 2)  # e.g., 97.5% for 95% confidence
        
        lower_bound = np.percentile(samples, lower_percentile)
        upper_bound = np.percentile(samples, upper_percentile)
        
        return lower_bound, upper_bound


# Base class for parametric distributions, handles parameter resolution and dependencies
class ParametricDistribution(Distribution):
    def __init__(self, distribution_type=None, **params):
        super().__init__(distribution_type=distribution_type)
        self.params = params

    def get_dependencies(self):
        deps = set()
        for value in self.params.values():
            if isinstance(value, StochasticVariable):
                deps.update(value.get_all_dependencies())
                deps.add(value)
        return deps

    def _resolve_params(self, context=None, size=1):
        if context is None:
            context = {}
        resolved_params = {}
        for key, value in self.params.items():
            if isinstance(value, StochasticVariable):
                if value not in context:
                    context[value] = value.sample(size=size, context=context)
                resolved_params[key] = context[value]
            else:
                resolved_params[key] = value
        return resolved_params


# Class for standard distributions, such as those from SciPy
class StandardDistribution(ParametricDistribution):
    def __init__(self, dist, **params):
        super().__init__(**params)
        self.dist = dist

        # Determine the distribution type without instantiation
        if isinstance(dist, rv_discrete):
            self.distribution_type = 'discrete'
        elif isinstance(dist, rv_continuous):
            self.distribution_type = 'continuous'
        else:
            # For distributions like dirichlet, set distribution_type in subclass
            self.distribution_type = None
            
    def sample(self, size=1, context=None):
        if context is None:
            context = {}

        # Resolve all parameters
        resolved_params = self._resolve_params(context=context, size=size)

        # Generate samples using resolved parameters
        dist_instance = self.dist(**{k: np.asarray(v).flatten() for k, v in resolved_params.items()})
        return dist_instance.rvs(size=size)

    def pdf(self, x, context=None):
        if self.distribution_type == 'discrete':
            raise ValueError("PDF is not defined for discrete distributions.")
        resolved_params = self._resolve_params(context=context)
        distribution_instance = self.dist(**resolved_params)
        if hasattr(distribution_instance, 'pdf'):
            return distribution_instance.pdf(x)
        else:
            raise NotImplementedError("PDF not available for this distribution.")

    def pmf(self, x, context=None):
        if self.distribution_type == 'continuous':
            raise ValueError("PMF is not defined for continuous distributions.")
        resolved_params = self._resolve_params(context=context)
        distribution_instance = self.dist(**resolved_params)
        if hasattr(distribution_instance, 'pmf'):
            return distribution_instance.pmf(x)
        else:
            raise NotImplementedError("PMF not available for this distribution.")

    def cdf(self, x, context=None):
        resolved_params = self._resolve_params(context=context)
        distribution_instance = self.dist(**resolved_params)
        if hasattr(distribution_instance, 'cdf'):
            return distribution_instance.cdf(x)
        else:
            raise NotImplementedError("CDF not available for this distribution.")

    # Inherit empirical methods from Distribution or override if needed


# Class for custom user-defined distributions
class CustomDistribution(ParametricDistribution):
    def __init__(self, func, domain=None, distribution_type='continuous', **params):
        super().__init__(distribution_type=distribution_type, **params)
        self.func = func
        self.domain = domain or (-np.inf, np.inf)

    def sample(self, size=1, context=None):
        # For custom distributions, you may need to implement specific sampling methods
        # Here, we'll use rejection sampling as an example for continuous distributions
        if self.distribution_type == 'continuous':
            samples = []
            max_pdf = self._get_max_pdf(context=context)
            while len(samples) < size:
                x = np.random.uniform(self.domain[0], self.domain[1], size=size)
                y = np.random.uniform(0, max_pdf, size=size)
                accepted = x[y < self.pdf(x, context=context)]
                samples.extend(accepted.tolist())
            return np.array(samples[:size])
        else:
            # For discrete distributions, you might need to define possible values
            raise NotImplementedError("Sampling for discrete custom distributions is not implemented.")

    def _get_max_pdf(self, context=None):
        # Estimate the maximum value of the PDF over the domain
        x = np.linspace(self.domain[0], self.domain[1], 1000)
        y = self.pdf(x, context=context)
        return np.max(y)

    def pdf(self, x, context=None):
        if self.distribution_type == 'discrete':
            raise ValueError("PDF is not defined for discrete distributions.")
        resolved_params = self._resolve_params(context=context)
        return self.func(x, **resolved_params)

    def pmf(self, x, context=None):
        if self.distribution_type == 'continuous':
            raise ValueError("PMF is not defined for continuous distributions.")
        resolved_params = self._resolve_params(context=context)
        return self.func(x, **resolved_params)

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at x.

        Parameters:
            - x (float): The value at which to evaluate the CDF.

        Returns:
            - CDF value at x.
        """
        if self.distribution_type == "continuous":
            a, b = self.domain  # Unpack the interval
            if x < a:
                return 0  # Below the domain
            elif x > b:
                return 1  # Above the domain
            else:
                # Integrate the PDF from a to x
                result, _ = quad(self.func, a, x)
                return result

        elif self.distribution_type == "discrete":
            if not isinstance(self.domain, list):
                raise ValueError("Domain must be a list for discrete distributions.")
            
            # Sum the PMF for values <= x
            cumulative_sum = sum(self.func(xi) for xi in self.domain if xi <= x)
            return cumulative_sum

        else:
            raise ValueError("Invalid distribution type. Must be 'continuous' or 'discrete'.")

    # Inherit empirical methods from Distribution class


# Class for creating mixture distributions
class MixtureDistribution(Distribution):
    def __init__(self, components, weights):
        """
        Represents a mixture distribution.

        Parameters:
        - components (list): List of Distribution instances (should be ScipyDistribution or compatible).
        - weights (list): List of weights (can include stochastic variables).
        """
        self.components = components
        self.weights = weights

        # Determine distribution type
        if all(c.distribution_type == 'continuous' for c in components):
            distribution_type = 'continuous'
        elif all(c.distribution_type == 'discrete' for c in components):
            distribution_type = 'discrete'
        else:
            distribution_type = 'mixed'

        # Store parameters for resolution
        self.params = {'components': components, 'weights': weights}
        super().__init__(distribution_type=distribution_type)

    def sample(self, size=1, context=None):
        if context is None:
            context = {}

        # Resolve weights
        resolved_weights = []
        for weight in self.weights:
            if isinstance(weight, StochasticVariable):
                weight_samples = weight.sample(size=size, context=context)
                resolved_weights.append(weight_samples)
            else:
                resolved_weights.append(np.full(size, weight))

        resolved_weights = np.column_stack(resolved_weights)
        # Normalize weights
        resolved_weights = resolved_weights / resolved_weights.sum(axis=1, keepdims=True)

        # Choose components based on weights
        cum_weights = np.cumsum(resolved_weights, axis=1)
        rand_vals = np.random.rand(size, 1)
        component_indices = (rand_vals < cum_weights).argmax(axis=1)

        # Sample from each component
        samples = np.zeros(size)
        for idx in range(len(self.components)):
            mask = component_indices == idx
            num_samples = np.sum(mask)
            if num_samples > 0:
                samples[mask] = self.components[idx].sample(size=num_samples, context=context).flatten()

        return samples

    def pdf(self, x, context=None):
        if self.distribution_type != 'continuous':
            raise NotImplementedError("PDF is only defined for continuous distributions.")

        x = np.array(x)
        pdf_values = np.zeros_like(x, dtype=float)

        # Resolve weights
        resolved_weights = []
        for weight in self.weights:
            if isinstance(weight, StochasticVariable):
                weight_sample = weight.sample(size=1, context=context).item()
                resolved_weights.append(weight_sample)
            else:
                resolved_weights.append(weight)
        resolved_weights = np.array(resolved_weights)
        resolved_weights /= resolved_weights.sum()

        # Compute weighted sum of PDFs
        for weight, component in zip(resolved_weights, self.components):
            pdf_values += weight * component.pdf(x, context=context)

        return pdf_values

    def pmf(self, x, context=None):
        if self.distribution_type != 'discrete':
            raise NotImplementedError("PMF is only defined for discrete distributions.")

        x = np.array(x)
        pmf_values = np.zeros_like(x, dtype=float)

        # Resolve weights
        resolved_weights = []
        for weight in self.weights:
            if isinstance(weight, StochasticVariable):
                weight_sample = weight.sample(size=1, context=context).item()
                resolved_weights.append(weight_sample)
            else:
                resolved_weights.append(weight)
        resolved_weights = np.array(resolved_weights)
        resolved_weights /= resolved_weights.sum()

        # Compute weighted sum of PMFs
        for weight, component in zip(resolved_weights, self.components):
            pmf_values += weight * component.pmf(x, context=context)

        return pmf_values

    def get_dependencies(self):
        deps = set()
        for component in self.components:
            deps.update(component.get_dependencies())
        for weight in self.weights:
            if isinstance(weight, StochasticVariable):
                deps.update(weight.get_all_dependencies())
                deps.add(weight)
        return deps




# Discrete distributions

class DiscreteUniformDistribution(StandardDistribution):
    def __init__(self, a, b):
        if not isinstance(a, int):
            raise ValueError(f"Argument 'a' must be an integer. Got a={a}")
        if not isinstance(b, int):
            raise ValueError(f"Argument 'b' must be an integer. Got b={b}")
        super().__init__(dist=randint, low=a, high=b + 1)

class BernoulliDistribution(StandardDistribution):
    def __init__(self, p):
        if p > 1 or p < 0:
            raise ValueError(f"Argument 'p' must be in the interval [0,1]. Got p={p}")
        super().__init__(dist=bernoulli, p=p)

class BinomialDistribution(StandardDistribution):
    def __init__(self, n, p):
        if (not isinstance(n, int)) or n < 0:
            raise ValueError(f"Argument 'n' must be a positive integer. Got n={n}")
        if p > 1 or p < 0:
            raise ValueError(f"Argument 'p' must be in the interval [0,1]. Got p={p}")
        super().__init__(dist=binom, n=n, p=p)

class GeometricDistribution(StandardDistribution):
    def __init__(self, p):
        if p > 1 or p < 0:
            raise ValueError(f"Argument 'p' must be in the interval [0,1]. Got p={p}")
        super().__init__(dist=geom, p=p)

class HypergeometricDistribution(StandardDistribution):
    def __init__(self, M, n, N):
        """
        Parameters:
            M: Total population size.
            n: Number of success states in the population.
            N: Number of draws.
        """
        if (not isinstance(n, int)) or n < 0:
            raise ValueError(f"Argument 'n' must be a positive integer. Got n={n}")
        if (not isinstance(N, int)) or N < 0:
            raise ValueError(f"Argument 'N' must be a positive integer. Got N={N}")
        if (not isinstance(M, int)) or M < 0:
            raise ValueError(f"Argument 'M' must be a positive integer. Got M={M}")
        super().__init__(dist=hypergeom, M=M, n=n, N=N)

class PoissonDistribution(StandardDistribution):
    def __init__(self, mu):
        super().__init__(dist=poisson, mu=mu)

class NegativeBinomialDistribution(StandardDistribution):
    def __init__(self, n, p):
        if (not isinstance(n, int)) or n < 0:
            raise ValueError(f"Argument 'n' must be a positive integer. Got n={n}")
        if p > 1 or p < 0:
            raise ValueError(f"Argument 'p' must be in the interval [0,1]. Got p={p}")
        super().__init__(dist=nbinom, n=n, p=p)

class MultinomialDistribution(StandardDistribution):
    def __init__(self, n, p):
        """
        Parameters:
            n: Number of trials.
            p: Sequence of probabilities. Must sum to 1.
        """
        if (not isinstance(n, int)) or n < 0:
            raise ValueError(f"Argument 'n' must be a positive integer. Got n={n}")
        if not np.isclose(sum(p), 1):
            raise ValueError(f"Argument 'p' must be a list of values summing to 1. Got sum(p)={sum(p)}")
        super().__init__(dist=multinomial, n=n, p=p)

    def sample(self, size=1, context=None):
        if context is None:
            context = {}
        # Resolve parameters
        resolved_params = self._resolve_params(context=context, size=size)
        n = resolved_params['n']
        p = resolved_params['p']
        if isinstance(p, StochasticVariable):
            p = p.sample(size=size, context=context)
        samples = multinomial.rvs(n=n, p=p, size=size)
        return samples

    # Multinomial distribution does not have pmf method in scipy.stats
    def pmf(self, x, context=None):
        resolved_params = self._resolve_params(context=context)
        n = resolved_params['n']
        p = resolved_params['p']
        x = np.atleast_2d(x)
        pmf_values = multinomial.pmf(x, n=n, p=p)
        return pmf_values


# Continuous distributions

class ContinuousUniformDistribution(StandardDistribution):
    def __init__(self, a, b):
        super().__init__(dist=uniform, loc=a, scale=b - a)

class ExponentialDistribution(StandardDistribution):
    def __init__(self, lambd=1):
        super().__init__(dist=expon, scale=1/lambd)

class NormalDistribution(StandardDistribution):
    def __init__(self, mu=0, sigma=1):
        super().__init__(dist=norm, loc=mu, scale=sigma)

    @classmethod
    def fit(cls, data):
        mu, sigma = norm.fit(data)
        return cls(mu=mu, sigma=sigma)

class LogNormalDistribution(StandardDistribution):
    def __init__(self, s, scale=np.exp(0)):
        """
        Parameters:
            s: Shape parameter (sigma of the underlying normal distribution).
            scale: Scale parameter (exp(mu) of the underlying normal distribution).
        """
        super().__init__(dist=lognorm, s=s, scale=scale)

    @classmethod
    def fit(cls, data):
        """
        Fit a log-normal distribution to the data and return an instance of LogNormalDistribution.

        Parameters:
            data (array-like): The data to fit.

        Returns:
            LogNormalDistribution: An instance with fitted parameters.
        """
        # Fix loc=0 for standard log-normal distribution
        s, loc, scale = lognorm.fit(data, floc=0)
        return cls(s=s, scale=scale)

class GammaDistribution(StandardDistribution):
    def __init__(self, shape, scale=1):
        """
        Parameters:
            shape: Shape parameter (also known as 'k' or 'alpha').
            scale: Scale parameter (theta).
        """
        super().__init__(dist=gamma, a=shape, scale=scale)

class ChiSquaredDistribution(StandardDistribution):
    def __init__(self, df):
        """
        Parameters:
            df: Degrees of freedom.
        """
        super().__init__(dist=chi2, df=df)

class RayleighDistribution(StandardDistribution):
    def __init__(self, scale=1):
        """
        Parameters:
            scale: Scale parameter (sigma).
        """
        super().__init__(dist=rayleigh, scale=scale)

class BetaDistribution(StandardDistribution):
    def __init__(self, a, b):
        """
        Parameters:
            a: Alpha parameter (>0).
            b: Beta parameter (>0).
        """
        super().__init__(dist=beta, a=a, b=b)

class CauchyDistribution(StandardDistribution):
    def __init__(self, x0=0, gamma=1):
        """
        Parameters:
            x0: Location parameter (median).
            gamma: Scale parameter (half-width at half-maximum).
        """
        super().__init__(dist=cauchy, loc=x0, scale=gamma)

class StandardArcsineDistribution(StandardDistribution):
    def __init__(self):
        """
        The standard arcsine distribution on the interval [0, 1].
        """
        super().__init__(dist=arcsine)

class DirichletDistribution(StandardDistribution):
    def __init__(self, alpha):
        """
        Parameters:
            alpha: Concentration parameters (array-like, all elements > 0).
        """
        super().__init__(dist=dirichlet, alpha=alpha)
        self.distribution_type = 'continuous'  # Set distribution type directly