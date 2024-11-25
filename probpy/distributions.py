# distributions.py

import numpy as np
from scipy.stats import rv_discrete, rv_continuous
from scipy.stats import norm, uniform, poisson, randint
from scipy.stats import kstest, chi2, anderson
from scipy.stats import norm, uniform, expon, poisson, randint, bernoulli, binom, geom, hypergeom, nbinom, multinomial, gamma, chi2, rayleigh, beta, cauchy, arcsine, dirichlet, rv_discrete, rv_continuous, gaussian_kde
from .core import StochasticVariable
from .constants import DEFAULT_STATISTICS_SAMPLE_SIZE


class Distribution:
    def __init__(self, distribution_type=None):
        self.distribution_type = distribution_type

    def sample(self, size=1, context=None):
        raise NotImplementedError("Sample method not implemented.")

    def get_dependencies(self):
        return set()


class ScipyDistribution(Distribution):
    def __init__(self, dist, **params):
        self.dist = dist
        self.params = params
        # Determine the distribution type
        if isinstance(dist, rv_discrete):
            super().__init__(distribution_type='discrete')
        elif isinstance(dist, rv_continuous):
            super().__init__(distribution_type='continuous')
        else:
            # Try to instantiate the distribution to check its type
            dist_instance = dist()
            if isinstance(dist_instance, rv_discrete):
                super().__init__(distribution_type='discrete')
            elif isinstance(dist_instance, rv_continuous):
                super().__init__(distribution_type='continuous')
            else:
                raise ValueError("The distribution must be an instance of rv_discrete or rv_continuous.")

    def sample(self, size=1, context=None):
        if context is None:
            context = {}
        resolved_params = self._resolve_params(context)
        distribution_instance = self.dist(**resolved_params)
        return distribution_instance.rvs(size=size)

    def get_dependencies(self):
        deps = set()
        for value in self.params.values():
            if isinstance(value, StochasticVariable):
                deps.update(value.get_all_dependencies())
                deps.add(value)
        return deps

    def _resolve_params(self, context=None):
        if context is None:
            context = {}
        resolved_params = {}
        for key, value in self.params.items():
            if isinstance(value, StochasticVariable):
                resolved_params[key] = value.sample(size=1, context=context)
            else:
                resolved_params[key] = value
        return resolved_params

    def pdf(self, x, context=None):
        resolved_params = self._resolve_params(context)
        distribution_instance = self.dist(**resolved_params)
        if hasattr(distribution_instance, 'pdf'):
            return distribution_instance.pdf(x)
        else:
            raise NotImplementedError("PDF not available for this distribution.")

    def pmf(self, x, context=None):
        resolved_params = self._resolve_params(context)
        distribution_instance = self.dist(**resolved_params)
        if hasattr(distribution_instance, 'pmf'):
            return distribution_instance.pmf(x)
        else:
            raise NotImplementedError("PMF not available for this distribution.")
        

    def cdf(self, x, context=None):
        resolved_params = self._resolve_params(context)
        distribution_instance = self.dist(**resolved_params)
        if hasattr(distribution_instance, 'cdf'):
            return distribution_instance.cdf(x)
        else:
            raise NotImplementedError("CDF not available for this distribution.")


class CustomDistribution(Distribution):
    def __init__(self, func, domain=None, size=DEFAULT_STATISTICS_SAMPLE_SIZE, distribution_type='continuous'):
        """
        Represents a user-defined distribution based on a lambda function.

        Parameters:
            - func (callable): A lambda function defining the distribution.
            - domain (tuple, list, or array): 
                - For continuous distributions: (low, high) tuple defining the domain.
                - For discrete distributions: List or array of discrete values.
            - size (int): The number of samples to use for empirical estimation (default: DEFAULT_STATISTICS_SAMPLE_SIZE).
            - distribution_type (str): Either 'continuous' or 'discrete' (default: 'continuous').
        """
        super().__init__(distribution_type=distribution_type)
        self.func = func
        self.domain = domain
        self.size = size

        # Generate samples for internal representation
        self.samples = self._generate_samples()

        # Continuous: Use KDE for PDF approximation
        if self.distribution_type == 'continuous':
            from scipy.stats import gaussian_kde
            self.kde = gaussian_kde(self.samples)
        # Discrete: Calculate empirical PMF
        elif self.distribution_type == 'discrete':
            unique, counts = np.unique(self.samples, return_counts=True)
            self.pmf_dict = dict(zip(unique, counts / counts.sum()))
        else:
            raise ValueError("distribution_type must be either 'continuous' or 'discrete'.")

    def _generate_samples(self):
        """
        Generates samples from the user-defined function over the specified domain.
        """
        if isinstance(self.domain, (list, np.ndarray)):
            # Discrete case: Use domain directly
            return np.array([self.func(x) for x in self.domain])
        elif isinstance(self.domain, tuple):
            # Continuous case: Generate points over the range (low, high)
            low, high = self.domain
            x_values = np.linspace(low, high, self.size)
            return np.array([self.func(x) for x in x_values])
        else:
            raise ValueError("Domain must be a tuple (low, high) or a list/array of discrete values.")

    def sample(self, size=1):
        """
        Samples from the user-defined distribution.

        Parameters:
            - size (int): Number of samples to generate.

        Returns:
            - Array of sampled values.
        """
        if self.distribution_type == 'continuous':
            samples = self.kde.resample(size*2).flatten()
            low, high = self.domain
            samples = samples[(samples >= low) & (samples <= high)]
            while len(samples) < size:
                new_samples = self.kde.resample(size).flatten()
                new_samples = new_samples[(new_samples >= low) & (new_samples <= high)]
                samples = np.concatenate((samples, new_samples))
            return samples[:size]
        elif self.distribution_type == 'discrete':
            # Discrete case: Use the pmf_dict to sample
            values, probabilities = zip(*self.pmf_dict.items())
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()  # Normalize probabilities
            return np.random.choice(values, size=size, p=probabilities)
        else:
            raise ValueError("Unsupported distribution type.")

    def pdf(self, x):
        """
        Probability density function for continuous distributions.
        """
        if self.distribution_type == 'continuous':
            return self.kde.evaluate(x)
        else:
            raise ValueError("PDF is not available for discrete distributions.")

    def pmf(self, x):
        """
        Probability mass function for discrete distributions.
        """
        if self.distribution_type == 'discrete':
            return np.array([self.pmf_dict.get(val, 0) for val in x])
        else:
            raise ValueError("PMF is not available for continuous distributions.")


class MixtureDistribution(Distribution):
    def __init__(self, components, weights):
        """
        Represents a mixture distribution.

        Parameters:
        - components (list): List of component distributions (subclasses of Distribution).
        - weights (list): List of weights for the components (must sum to 1).
        """
        # Determine the distribution type based on components
        if all(component.distribution_type == "continuous" for component in components):
            distribution_type = "continuous"
        elif all(component.distribution_type == "discrete" for component in components):
            distribution_type = "discrete"
        else:
            distribution_type = "mixed"

        super().__init__(distribution_type=distribution_type)

        # Validate components and weights
        assert len(components) == len(weights), "Components and weights must have the same length."
        assert np.isclose(sum(weights), 1.0), "Weights must sum to 1."

        self.components = components
        self.weights = weights

    def sample(self, size=1, context=None):
        """
        Samples from the mixture distribution.

        Parameters:
        - size (int): Number of samples to generate.
        - context (dict): Context for dependency resolution (not used here).

        Returns:
        - numpy.ndarray: Samples from the mixture distribution.
        """
        samples = []
        # Randomly choose components based on weights
        component_choices = np.random.choice(len(self.components), size=size, p=self.weights)
        for i, component in enumerate(self.components):
            count = np.sum(component_choices == i)
            if count > 0:
                samples.append(component.sample(size=count, context=context))
        return np.concatenate(samples)

    def pdf(self, x):
        """
        Evaluates the Probability Density Function (PDF) of the mixture distribution.

        Parameters:
        - x (float or array-like): Points at which to evaluate the PDF.

        Returns:
        - numpy.ndarray: The PDF values.
        """
        if self.distribution_type != "continuous":
            raise NotImplementedError("PDF is only defined for continuous distributions.")
        
        x = np.array(x)
        pdf_values = np.zeros_like(x, dtype=float)
        for component, weight in zip(self.components, self.weights):
            pdf_values += weight * component.pdf(x)
        return pdf_values

    def pmf(self, x):
        """
        Evaluates the Probability Mass Function (PMF) of the mixture distribution.

        Parameters:
        - x (int or array-like): Points at which to evaluate the PMF.

        Returns:
        - numpy.ndarray: The PMF values.
        """
        if self.distribution_type != "discrete":
            raise NotImplementedError("PMF is only defined for discrete distributions.")
        
        x = np.array(x)
        pmf_values = np.zeros_like(x, dtype=float)
        for component, weight in zip(self.components, self.weights):
            pmf_values += weight * component.pmf(x)
        return pmf_values

    def get_all_dependencies(self):
        """
        Recursively collects dependencies from all component distributions.

        Returns:
        - set: A set of dependencies across all components.
        """
        dependencies = set()
        for component in self.components:
            dependencies.update(component.get_dependencies())
        return dependencies




class BernoulliDistribution(ScipyDistribution):
    def __init__(self, p):
        """
        Represents a Bernoulli distribution with success probability p.
        
        Parameters:
        - p (float): Probability of success (0 <= p <= 1).
        """
        super().__init__(dist=bernoulli, p=p)




class BinomialDistribution(ScipyDistribution):
    def __init__(self, n, p):
        """
        Represents a Binomial distribution with n trials and success probability p.
        
        Parameters:
        - n (int): Number of trials.
        - p (float): Probability of success (0 <= p <= 1).
        """
        super().__init__(dist=binom, n=n, p=p)




class GeometricDistribution(ScipyDistribution):
    def __init__(self, p):
        """
        Represents a Geometric distribution with success probability p.
        
        Parameters:
        - p (float): Probability of success (0 <= p <= 1).
        """
        super().__init__(dist=geom, p=p)





class HypergeometricDistribution(ScipyDistribution):
    def __init__(self, M, n, N):
        """
        Represents a Hypergeometric distribution.
        
        Parameters:
        - M (int): Total population size.
        - n (int): Number of successes in the population.
        - N (int): Number of draws.
        """
        super().__init__(dist=hypergeom, M=M, n=n, N=N)





class PoissonDistribution(ScipyDistribution):
    def __init__(self, mu):
        """
        Represents a Poisson distribution with mean mu.
        
        Parameters:
        - mu (float): Mean number of events.
        """
        super().__init__(dist=poisson, mu=mu)





class NegativeBinomialDistribution(ScipyDistribution):
    def __init__(self, n, p):
        """
        Represents a Negative Binomial distribution.
        
        Parameters:
        - n (int): Number of successes.
        - p (float): Probability of success (0 <= p <= 1).
        """
        super().__init__(dist=nbinom, n=n, p=p)








class MultinomialDistribution(ScipyDistribution):
    def __init__(self, n, p):
        """
        Represents a Multinomial distribution.
        
        Parameters:
        - n (int): Number of trials.
        - p (list): List of probabilities for each outcome.
        """
        super().__init__(dist=multinomial, n=n, p=p)







class ExponentialDistribution(ScipyDistribution):
    def __init__(self, lambd):
        """
        Represents an Exponential distribution with rate parameter lambda.
        
        Parameters:
        - lambd (float): Rate parameter.
        """
        super().__init__(dist=expon, scale=1 / lambd)




class NormalDistribution(ScipyDistribution):
    def __init__(self, mu=0, sigma=1):
        """
        Represents a Normal distribution with mean mu and standard deviation sigma.
        
        Parameters:
        - mu (float): Mean.
        - sigma (float): Standard deviation.
        """
        super().__init__(dist=norm, loc=mu, scale=sigma)

    @classmethod
    def fit(cls, data):
        """
        Fits a normal distribution to the given data using MLE.

        Parameters:
            - data (list or numpy.ndarray): Observed data to fit.

        Returns:
            - NormalDistribution: A new NormalDistribution instance with the fitted parameters.
        """
        # Estimate parameters using Maximum Likelihood Estimation (MLE)
        mu, sigma = norm.fit(data)  # Scipy's MLE fit
        return cls(mu=mu, sigma=sigma)
    
    def goodness_of_fit(self, data, test="ks"):
        """
        Performs a goodness-of-fit test.

        Parameters:
        - data (array-like): Observed data to compare against the distribution.
        - test (str): The goodness-of-fit test to perform ('ks', 'chi2', 'anderson').

        Returns:
        - dict: Results of the goodness-of-fit test.
        """
        if test == "ks":
            # Kolmogorov-Smirnov Test
            stat, p_value = kstest(data, self.cdf)
            return {"test": "Kolmogorov-Smirnov", "statistic": stat, "p_value": p_value}

        elif test == "chi2":
            raise ValueError("Chi-Square test is not applicable for continuous distributions.")

        elif test == "anderson":
            result = anderson(data, dist="norm")
            return {"test": "Anderson-Darling", "statistic": result.statistic, "critical_values": result.critical_values}

        else:
            raise ValueError("Unsupported test type. Use 'ks', 'chi2', or 'anderson'.")




class GammaDistribution(ScipyDistribution):
    def __init__(self, shape, scale=1):
        """
        Represents a Gamma distribution.
        
        Parameters:
        - shape (float): Shape parameter (k).
        - scale (float): Scale parameter (θ).
        """
        super().__init__(dist=gamma, a=shape, scale=scale)





class ChiSquaredDistribution(ScipyDistribution):
    def __init__(self, df):
        """
        Represents a Chi-Squared distribution with degrees of freedom df.
        
        Parameters:
        - df (int): Degrees of freedom.
        """
        super().__init__(dist=chi2, df=df)






class RayleighDistribution(ScipyDistribution):
    def __init__(self, scale=1):
        """
        Represents a Rayleigh distribution.
        
        Parameters:
        - scale (float): Scale parameter.
        """
        super().__init__(dist=rayleigh, scale=scale)





class BetaDistribution(ScipyDistribution):
    def __init__(self, alpha, beta):
        """
        Represents a Beta distribution.
        
        Parameters:
        - alpha (float): Alpha parameter.
        - beta (float): Beta parameter.
        """
        super().__init__(dist=beta, a=alpha, b=beta)





class CauchyDistribution(ScipyDistribution):
    def __init__(self, loc=0, scale=1):
        """
        Represents a Cauchy distribution.
        
        Parameters:
        - loc (float): Location parameter.
        - scale (float): Scale parameter.
        """
        super().__init__(dist=cauchy, loc=loc, scale=scale)





class StandardArcsineDistribution(ScipyDistribution):
    def __init__(self):
        """
        Represents a Standard Arcsine distribution.
        """
        super().__init__(dist=arcsine)






class DirichletDistribution(ScipyDistribution):
    def __init__(self, alpha):
        """
        Represents a Dirichlet distribution.
        
        Parameters:
        - alpha (list): Concentration parameters for the distribution.
        """
        super().__init__(dist=dirichlet, alpha=alpha)




class ContinuousUniformDistribution(ScipyDistribution):
    def __init__(self, a=0, b=1):
        """
        Represents a continuous uniform distribution over the interval [a, b].
        
        Parameters:
        - a (float): Lower bound of the interval.
        - b (float): Upper bound of the interval.
        """
        super().__init__(dist=uniform, loc=a, scale=b - a)






class DiscreteUniformDistribution(ScipyDistribution):
    def __init__(self, a, b):
        """
        Represents a discrete uniform distribution over integers [a, b].
        
        Parameters:
        - a (int): Lower bound (inclusive).
        - b (int): Upper bound (inclusive).
        """
        super().__init__(dist=randint, low=a, high=b + 1)




