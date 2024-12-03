# test_probpy.py

import unittest
import numpy as np
from scipy.stats import (
    norm, expon, uniform, binom, poisson, beta, gamma, chi2, kstest
)
from probpy.core import (
    StochasticVariable,
    StochasticVector,
    StochasticMatrix,
    apply,
    probability,
)
from probpy.distributions import (
    NormalDistribution,
    ExponentialDistribution,
    ContinuousUniformDistribution,
    BinomialDistribution,
    BernoulliDistribution,
    PoissonDistribution,
    BetaDistribution,
    GammaDistribution,
    LogNormalDistribution,
    DiscreteUniformDistribution,
    DirichletDistribution,
    ChiSquaredDistribution,
)
from probpy.transformations import (
    exp, log, sqrt, square, power,
    sin, cos, tan, arcsin, arccos, arctan,
    sinh, cosh, tanh, arcsinh, arccosh, arctanh,
)
from probpy.plots import plot_distribution, plot_dependency_graph
from probpy.monte_carlo import monte_carlo_simulate, summarize_simulation
import warnings

# Suppress warnings from statistical tests
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants for sample sizes
SMALL_SAMPLE = 100
MEDIUM_SAMPLE = 1000
LARGE_SAMPLE = 1000000


class TestDistributions(unittest.TestCase):

    def test_normal_distribution_properties(self):
        StochasticVariable.delete_all_instances()
        mu = 5
        sigma = 2
        dist = NormalDistribution(mu=mu, sigma=sigma)
        samples = dist.sample(size=LARGE_SAMPLE)
        self.assertAlmostEqual(np.mean(samples), mu, delta=0.05)
        self.assertAlmostEqual(np.std(samples), sigma, delta=0.05)

    def test_exponential_distribution_properties(self):
        StochasticVariable.delete_all_instances()
        lambd = 0.5
        dist = ExponentialDistribution(lambd=lambd)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = 1 / lambd
        expected_variance = 1 / (lambd ** 2)
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.05)
        self.assertAlmostEqual(np.var(samples), expected_variance, delta=0.1)

    def test_binomial_distribution_properties(self):
        StochasticVariable.delete_all_instances()
        n = 20
        p = 0.7
        dist = BinomialDistribution(n=n, p=p)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = n * p
        expected_variance = n * p * (1 - p)
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.1)
        self.assertAlmostEqual(np.var(samples), expected_variance, delta=0.1)

    def test_poisson_distribution_properties(self):
        StochasticVariable.delete_all_instances()
        mu = 4
        dist = PoissonDistribution(mu=mu)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = mu
        expected_variance = mu
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.05)
        self.assertAlmostEqual(np.var(samples), expected_variance, delta=0.05)

    def test_beta_distribution_properties(self):
        StochasticVariable.delete_all_instances()
        a = 2
        b = 3
        dist = BetaDistribution(a=a, b=b)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = a / (a + b)
        expected_variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.01)
        self.assertAlmostEqual(np.var(samples), expected_variance, delta=0.01)

    def test_gamma_distribution_properties(self):
        StochasticVariable.delete_all_instances()
        shape = 5
        scale = 2
        dist = GammaDistribution(shape=shape, scale=scale)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = shape * scale
        expected_variance = shape * (scale ** 2)
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.1)
        self.assertAlmostEqual(np.var(samples), expected_variance, delta=0.2)



    def test_dirichlet_distribution(self):
        StochasticVariable.delete_all_instances()
        alpha = [2, 3, 5]
        dist = DirichletDistribution(alpha=alpha)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = np.array(alpha) / np.sum(alpha)
        sample_mean = np.mean(samples, axis=0)
        self.assertTrue(np.allclose(sample_mean, expected_mean, atol=0.01))

class TestTransformations(unittest.TestCase):

    def test_trigonometric_functions(self):
        StochasticVariable.delete_all_instances()
        x = StochasticVariable(NormalDistribution(mu=0, sigma=np.pi / 4), name='X')
        sin_x = sin(x)
        cos_x = cos(x)
        tan_x = tan(x)
        sin_samples = sin_x.sample(size=LARGE_SAMPLE)
        cos_samples = cos_x.sample(size=LARGE_SAMPLE)
        tan_samples = tan_x.sample(size=LARGE_SAMPLE)
        self.assertTrue(np.all(np.abs(sin_samples) <= 1))
        self.assertTrue(np.all(np.abs(cos_samples) <= 1))
        # Tan function will have large values near pi/2, check for finiteness
        self.assertTrue(np.all(np.isfinite(tan_samples)))

    def test_inverse_trigonometric_functions(self):
        StochasticVariable.delete_all_instances()
        x = StochasticVariable(ContinuousUniformDistribution(a=-1, b=1), name='X')
        arcsin_x = arcsin(x)
        arccos_x = arccos(x)
        arctan_x = arctan(x)
        arcsin_samples = arcsin_x.sample(size=LARGE_SAMPLE)
        arccos_samples = arccos_x.sample(size=LARGE_SAMPLE)
        arctan_samples = arctan_x.sample(size=LARGE_SAMPLE)
        # Check that arcsin and arccos outputs are within correct ranges
        self.assertTrue(np.all((arcsin_samples >= -np.pi / 2) & (arcsin_samples <= np.pi / 2)))
        self.assertTrue(np.all((arccos_samples >= 0) & (arccos_samples <= np.pi)))
        self.assertTrue(np.all(np.isfinite(arctan_samples)))

    def test_hyperbolic_functions(self):
        StochasticVariable.delete_all_instances()
        x = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X')
        sinh_x = sinh(x)
        cosh_x = cosh(x)
        tanh_x = tanh(x)
        sinh_samples = sinh_x.sample(size=LARGE_SAMPLE)
        cosh_samples = cosh_x.sample(size=LARGE_SAMPLE)
        tanh_samples = tanh_x.sample(size=LARGE_SAMPLE)
        # No specific range, just check for finiteness
        self.assertTrue(np.all(np.isfinite(sinh_samples)))
        self.assertTrue(np.all(cosh_samples >= 1))  # cosh(x) >= 1
        self.assertTrue(np.all(np.abs(tanh_samples) <= 1))

    def test_logarithmic_functions(self):
        StochasticVariable.delete_all_instances()
        x = StochasticVariable(ExponentialDistribution(lambd=1), name='X')
        log_x = log(x)
        log_samples = log_x.sample(size=LARGE_SAMPLE)
        # Since X > 0, log(X) should be defined
        self.assertTrue(np.all(np.isfinite(log_samples)))

    def test_power_functions(self):
        StochasticVariable.delete_all_instances()
        x = StochasticVariable(ContinuousUniformDistribution(a=0, b=10), name='X')
        sqrt_x = sqrt(x)
        square_x = square(x)
        power_x = power(x, 3)
        sqrt_samples = sqrt_x.sample(size=LARGE_SAMPLE)
        square_samples = square_x.sample(size=LARGE_SAMPLE)
        power_samples = power_x.sample(size=LARGE_SAMPLE)
        self.assertTrue(np.all(sqrt_samples >= 0))
        self.assertTrue(np.all(square_samples >= 0))
        self.assertTrue(np.all(power_samples >= 0))

class TestDependencyStructures(unittest.TestCase):

    def test_markov_chain(self):
        StochasticVariable.delete_all_instances()
        # Define a simple Markov chain X -> Y -> Z
        X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X')
        Y = StochasticVariable(NormalDistribution(mu=X, sigma=1), name='Y')
        Z = StochasticVariable(NormalDistribution(mu=Y, sigma=1), name='Z')

        context = {}
        X_samples = X.sample(size=LARGE_SAMPLE, context=context)
        Y_samples = Y.sample(size=LARGE_SAMPLE, context=context)
        Z_samples = Z.sample(size=LARGE_SAMPLE, context=context)

        # Check conditional expectations
        self.assertAlmostEqual(np.mean(Y_samples - X_samples), 0, delta=0.05)
        self.assertAlmostEqual(np.mean(Z_samples - Y_samples), 0, delta=0.05)

    def test_bayesian_network(self):
        StochasticVariable.delete_all_instances()
        # Define a simple Bayesian network A -> C <- B
        A = StochasticVariable(BernoulliDistribution(p=0.6), name='A')
        B = StochasticVariable(BernoulliDistribution(p=0.7), name='B')

        C = apply(lambda a, b: a & b, A, B, name='C')

        context = {}
        A_samples = A.sample(size=LARGE_SAMPLE, context=context)
        B_samples = B.sample(size=LARGE_SAMPLE, context=context)
        C_samples = C.sample(size=LARGE_SAMPLE, context=context)

        expected_p_C = 0.6 * 0.7
        actual_p_C = np.mean(C_samples)
        self.assertAlmostEqual(actual_p_C, expected_p_C, delta=0.01)

class TestStatisticalProperties(unittest.TestCase):


    def test_sum_of_variables(self):
        StochasticVariable.delete_all_instances()
        import numpy as np
        from functools import reduce
        import operator

        # Assuming all necessary classes are already defined and imported

        n = 10

        # Define the stochastic variables
        M = StochasticVariable(ContinuousUniformDistribution(-2, 3), name="M")
        V = StochasticVariable(ExponentialDistribution(lambd=0.7), name="V")

        mu_M = (-2 + 3) / 2
        var_M = ((3 - (-2)) ** 2) / 12

        mu_V = 1 / 0.7

        # Correctly pass the standard deviation (sqrt(V)) instead of variance (V)

        Xs = [StochasticVariable(NormalDistribution(M, apply(lambda v: np.sqrt(v), V)), name=f"X_{i}") for i in range(n)]

        # Sum the stochastic variables
        X = sum(Xs)
        X.name = "X"

        # Sample and compute statistics
        actual_mean = X.mean(size=100000)
        actual_var = X.var(size=100000)

        expected_mean = n * mu_M
        expected_variance = n * mu_V + (n ** 2) * var_M

        # print(f"Expected mean: {expected_mean} vs. Observed mean: {actual_mean}")
        # print(f"Expected variance: {expected_variance} vs. Observed variance: {actual_var}")

        # Directly simulate M and V
        M_samples = np.random.uniform(-2, 3, size=100000)
        V_samples = np.random.exponential(scale=1/0.7, size=100000)

        # Simulate X_i with correct standard deviation
        X_samples = []
        for _ in range(n):
            X_samples.append(np.random.normal(M_samples, np.sqrt(V_samples)))

        # Sum X_i to get X
        X_direct = np.sum(X_samples, axis=0)

        # Compute mean and variance
        actual_mean_direct = np.mean(X_direct)
        actual_var_direct = np.var(X_direct)

        # print(f"Direct Simulation - Mean: {actual_mean_direct}, Variance: {actual_var_direct}")

        direct_dev_mean = (actual_mean_direct - expected_mean) / expected_mean
        direct_dev_variance = (actual_var_direct - expected_variance) / expected_variance

        simulation_dev_mean = (actual_mean - expected_mean) / expected_mean
        simulation_dev_variance = (actual_var - expected_variance) / expected_variance

        self.assertAlmostEqual(direct_dev_mean, simulation_dev_mean, delta=0.1)
        self.assertAlmostEqual(direct_dev_variance, simulation_dev_variance, delta=0.1)


    def test_covariance(self):
        StochasticVariable.delete_all_instances()
        # Test covariance between two dependent variables
        X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X')
        Y = 2 * X + StochasticVariable(NormalDistribution(mu=0, sigma=1), name='Epsilon')
        context = {}
        X_samples = X.sample(size=LARGE_SAMPLE, context=context)
        Y_samples = Y.sample(size=LARGE_SAMPLE, context=context)
        covariance = np.cov(X_samples, Y_samples)[0, 1]
        expected_covariance = 2  # Since Cov(X, 2X) = 2 * Var(X)
        self.assertAlmostEqual(covariance, expected_covariance, delta=0.1)

    def test_correlation_coefficient(self):
        StochasticVariable.delete_all_instances()
        # Test correlation coefficient between two variables
        X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X')
        Y = -3 * X + StochasticVariable(NormalDistribution(mu=0, sigma=1), name='Epsilon')
        context = {}
        X_samples = X.sample(size=LARGE_SAMPLE, context=context)
        Y_samples = Y.sample(size=LARGE_SAMPLE, context=context)
        correlation = np.corrcoef(X_samples, Y_samples)[0, 1]
        expected_correlation = -3 / np.sqrt(1 + 9)  # -3 / sqrt(1 + 9)
        self.assertAlmostEqual(correlation, expected_correlation, delta=0.05)

class TestAdvancedUsage(unittest.TestCase):

    def test_joint_distribution_sampling(self):
        StochasticVariable.delete_all_instances()
        # Test sampling from joint distribution of dependent variables
        X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X')
        Y = X + StochasticVariable(NormalDistribution(mu=0, sigma=1), name='Epsilon')
        Z = X - Y
        context = {}
        samples = np.column_stack([X.sample(size=LARGE_SAMPLE, context=context),
                                   Y.sample(size=LARGE_SAMPLE, context=context),
                                   Z.sample(size=LARGE_SAMPLE, context=context)])
        # Test that Z = X - Y holds
        self.assertTrue(np.allclose(samples[:, 2], samples[:, 0] - samples[:, 1]))

    def test_sampling_with_constraints(self):
        StochasticVariable.delete_all_instances()
        # Test sampling with constraints using rejection sampling
        X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X')
        condition = lambda x: x > 0  # Only positive samples
        samples = []
        while len(samples) < LARGE_SAMPLE:
            x_samples = X.sample(size=LARGE_SAMPLE)
            accepted = x_samples[x_samples > 0]
            samples.extend(accepted)
        samples = np.array(samples[:LARGE_SAMPLE])
        # Check that all samples are positive
        self.assertTrue(np.all(samples > 0))

    def test_empirical_cdf(self):
        StochasticVariable.delete_all_instances()
        # Test empirical cumulative distribution function
        X = StochasticVariable(ExponentialDistribution(lambd=1), name='X')
        samples = X.sample(size=LARGE_SAMPLE)
        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, LARGE_SAMPLE + 1) / LARGE_SAMPLE
        # Compare with theoretical CDF
        theoretical_cdf = 1 - np.exp(-sorted_samples)
        # Compute maximum difference (Kolmogorov-Smirnov statistic)
        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        # For large samples, ks_statistic should be small
        self.assertLess(ks_statistic, 0.01)

class TestMonteCarloSimulation(unittest.TestCase):

    def test_option_pricing(self):
        StochasticVariable.delete_all_instances()
        # Use Monte Carlo simulation to price a European call option
        S0 = 100  # Initial stock price
        K = 105   # Strike price
        T = 1     # Time to maturity in years
        r = 0.05  # Risk-free rate
        sigma = 0.2  # Volatility

        def model(Z):
            ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
            payoff = np.maximum(ST - K, 0)
            return np.exp(-r * T) * payoff

        Z = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='Z')
        results = monte_carlo_simulate(model, [Z], trials=LARGE_SAMPLE)
        estimated_price = np.mean(results)

        # Black-Scholes formula for European call option price
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        self.assertAlmostEqual(estimated_price, bs_price, delta=0.1)

    def test_integral_estimation(self):
        StochasticVariable.delete_all_instances()
        # Estimate the value of an integral using Monte Carlo simulation
        # Integral of sin(x) from 0 to pi
        def model(x):
            return np.sin(x)

        X = StochasticVariable(ContinuousUniformDistribution(a=0, b=np.pi), name='X')
        results = monte_carlo_simulate(model, [X], trials=LARGE_SAMPLE)
        estimated_integral = (np.pi - 0) * np.mean(results)
        actual_integral = 2  # Known value of the integral
        self.assertAlmostEqual(estimated_integral, actual_integral, delta=0.01)

    def test_multidimensional_integration(self):
        StochasticVariable.delete_all_instances()
        # Estimate the volume of a 4-dimensional unit sphere
        def model(x1, x2, x3, x4):
            return (x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2) <= 1

        variables = [
            StochasticVariable(ContinuousUniformDistribution(a=-1, b=1), name=f'X{i}') for i in range(4)
        ]
        results = monte_carlo_simulate(model, variables, trials=LARGE_SAMPLE)
        estimated_volume = (2 ** 4) * np.mean(results)
        # Actual volume of a 4D unit sphere: (π^2) / 2
        actual_volume = (np.pi ** 2) / 2
        self.assertAlmostEqual(estimated_volume, actual_volume, delta=0.1)

class TestEdgeCases(unittest.TestCase):

    def test_extreme_parameter_values(self):
        StochasticVariable.delete_all_instances()
        # Test distributions with extreme parameter values
        dist = NormalDistribution(mu=0, sigma=1e-10)
        samples = dist.sample(size=LARGE_SAMPLE)
        # Samples should be very close to 0
        self.assertTrue(np.allclose(samples, 0, atol=1e-9))

    def test_zero_variance(self):
        StochasticVariable.delete_all_instances()
        # Test behavior when variance is zero
        dist = NormalDistribution(mu=5, sigma=0)
        samples = dist.sample(size=LARGE_SAMPLE)
        self.assertTrue(np.all(samples == 5))

    def test_large_numbers(self):
        StochasticVariable.delete_all_instances()
        # Test with very large numbers
        lambd = 1e-5
        dist = ExponentialDistribution(lambd=lambd)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = 1 / lambd
        self.assertTrue(np.all(samples >= 0))
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=expected_mean * 0.05)

    def test_small_probabilities(self):
        StochasticVariable.delete_all_instances()
        # Test binomial distribution with small probability
        n = 1000
        p = 1e-5
        dist = BinomialDistribution(n=n, p=p)
        samples = dist.sample(size=LARGE_SAMPLE)
        expected_mean = n * p
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.1)

class TestGoodnessOfFit(unittest.TestCase):

    def test_lognormal_distribution_fit(self):
        StochasticVariable.delete_all_instances()
        # Generate data from a lognormal distribution
        mu = 0
        sigma = 1
        dist = LogNormalDistribution(s=sigma, scale=np.exp(mu))
        samples = dist.sample(size=LARGE_SAMPLE)
        # Fit the distribution to the samples
        fitted_dist = LogNormalDistribution.fit(samples)
        self.assertAlmostEqual(fitted_dist.params['s'], sigma, delta=0.1)
        self.assertAlmostEqual(np.log(fitted_dist.params['scale']), mu, delta=0.1)

class TestDependencyGraph(unittest.TestCase):

    def test_plot_dependency_graph_with_vectors(self):
        StochasticVariable.delete_all_instances()
        # Create stochastic variables and vectors
        X1 = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X1')
        X2 = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X2')
        V1 = StochasticVector(X1, X2, name='V1')

        Y = X1 + X2
        Y.name = 'Y'
        V2 = V1 * 2
        V2.name = 'V2'
        # Plot the dependency graph

        V3 = StochasticVector(X1, X2, V2.norm(), name="V3")

        try:
            plot_dependency_graph(V3, V1, V2, Y, title="Dependency Graph with Vectors", depth=1)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)


class TestStochasticMatrix(unittest.TestCase):

    def test_matrix_operations(self):
        StochasticVariable.delete_all_instances()
        # Define a 2x2 stochastic matrix
        a = StochasticVariable(NormalDistribution(mu=1, sigma=0.1), name='a')
        b = StochasticVariable(NormalDistribution(mu=2, sigma=0.1), name='b')
        c = StochasticVariable(NormalDistribution(mu=3, sigma=0.1), name='c')
        d = StochasticVariable(NormalDistribution(mu=4, sigma=0.1), name='d')

        matrix1 = StochasticMatrix([[a, b], [c, d]], name='Matrix1')
        matrix2 = matrix1.transpose()

        # Test matrix multiplication
        result_matrix = matrix1 @ matrix2
        samples = result_matrix.sample(size=1000)
        self.assertEqual(samples.shape, (1000, 2, 2))

        print(matrix1)

        # Test that the result has the expected mean values
        expected_means = np.array([[a.mean() * a.mean() + b.mean() * b.mean(),
                                    a.mean() * c.mean() + b.mean() * d.mean()],
                                   [c.mean() * a.mean() + d.mean() * b.mean(),
                                    c.mean() * c.mean() + d.mean() * d.mean()]])
        sample_means = np.mean(samples, axis=0)
        np.testing.assert_allclose(sample_means, expected_means, atol=0.1)

if __name__ == "__main__":
    unittest.main()