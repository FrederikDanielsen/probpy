# examples.py

import numpy as np
from probpy.core import (
    StochasticVariable,
    StochasticVector,
    apply,
    probability,
)
from probpy.distributions import (
    CustomDistribution,
    MixtureDistribution,
    NormalDistribution,
    ExponentialDistribution,
    ContinuousUniformDistribution,
    DiscreteUniformDistribution,
    BinomialDistribution,
    PoissonDistribution,
    BetaDistribution,
    GammaDistribution,
    ChiSquaredDistribution,
    DirichletDistribution,
)
from probpy.transformations import (
    exp, log, sqrt, sin, cos, tan, arcsin, arccos, arctan, abs_
)
from probpy.plots import plot_distribution, plot_dependency_graph
from probpy.monte_carlo import monte_carlo_simulate, summarize_simulation, plot_simulation
import matplotlib.pyplot as plt

# Set constants for sample sizes
SMALL_SAMPLE = 100
MEDIUM_SAMPLE = 1000
LARGE_SAMPLE = 100000

print("Section 1: Basic Distributions")

# Normal Distribution
mu, sigma = 0, 1
normal_dist = NormalDistribution(mu=mu, sigma=sigma)
X = StochasticVariable(distribution=normal_dist, name='X_Normal')

# Sample and plot
samples = X.sample(size=LARGE_SAMPLE)
print(f"Normal Distribution Mean: {np.mean(samples):.4f}, Std Dev: {np.std(samples):.4f}")
plot_distribution(X, num_samples=1000, bins=30, title='Normal Distribution')

# Exponential Distribution
lambd = 1
exp_dist = ExponentialDistribution(lambd=lambd)
Y = StochasticVariable(distribution=exp_dist, name='Y_Exponential')

# Sample and plot
samples = Y.sample(size=LARGE_SAMPLE)
print(f"Exponential Distribution Mean: {np.mean(samples):.4f}")
plot_distribution(Y, num_samples=1000, bins=30, title='Exponential Distribution')

# ---------------------------
# Section 2: Stochastic Variables and Operations
# ---------------------------

print("\nSection 2: Stochastic Variables and Operations")

# Arithmetic Operations
A = StochasticVariable(distribution=NormalDistribution(mu=5, sigma=2), name='A')
B = StochasticVariable(distribution=NormalDistribution(mu=3, sigma=1), name='B')

# Addition
C = A + B
C.name = 'C = A + B'
samples = C.sample(size=LARGE_SAMPLE)
print(f"C Mean: {np.mean(samples):.4f}, Expected: {5 + 3}")

# Multiplication
D = A * B
D.name = 'D = A * B'
samples = D.sample(size=LARGE_SAMPLE)
print(f"D Mean: {np.mean(samples):.4f}, Expected: {5 * 3}")

# Custom Function Application
def custom_function(a_samples, b_samples):
    return np.sin(a_samples) + np.cos(b_samples)

E = apply(custom_function, A, B, name='E = sin(A) + cos(B)')
samples = E.sample(size=LARGE_SAMPLE)
print(f"E Mean: {np.mean(samples):.4f}")

# ---------------------------
# Section 3: Transformations
# ---------------------------

print("\nSection 3: Transformations")

# Exponential Transformation
F = exp(A)
F.name = 'F = exp(A)'
samples = F.sample(size=LARGE_SAMPLE)
print(f"F Mean: {np.mean(samples):.4f}")
plot_distribution(F, num_samples=1000, bins=30, title='Exponential Transformation')

# Logarithmic Transformation
G = log(Y)
G.name = 'G = log(Y)'
samples = G.sample(size=LARGE_SAMPLE)
print(f"G Mean: {np.mean(samples):.4f}")
plot_distribution(G, num_samples=1000, bins=30, title='Logarithmic Transformation')

# ---------------------------
# Section 4: Stochastic Vectors
# ---------------------------

print("\nSection 4: Stochastic Vectors")

# Create a stochastic vector
X1 = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X1')
X2 = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X2')
vector = StochasticVector(X1, X2, name='Vector_X')

# Compute Norm
vector_norm = vector.norm(p=2)
vector_norm.name = 'Vector Norm'
samples = vector_norm.sample(size=LARGE_SAMPLE)
expected_mean = np.sqrt(2) * np.exp(np.log(np.sqrt(np.pi / 2)))
print(f"Vector Norm Mean: {np.mean(samples):.4f}, Expected: {expected_mean:.4f}")

# Dot Product
Y1 = StochasticVariable(NormalDistribution(mu=1, sigma=1), name='Y1')
Y2 = StochasticVariable(NormalDistribution(mu=2, sigma=1), name='Y2')
vector_Y = StochasticVector(Y1, Y2, name='Vector_Y')

dot_product = vector.dot(vector_Y)
dot_product.name = 'Dot Product'
samples = dot_product.sample(size=LARGE_SAMPLE)
expected_mean = (0 * 1) + (0 * 2)
print(f"Dot Product Mean: {np.mean(samples):.4f}, Expected: {expected_mean:.4f}")

# Cross Product (3D vectors)
X3 = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X3')
vector_3D = StochasticVector(X1, X2, X3, name='Vector_X3D')
vector_Y3D = StochasticVector(Y1, Y2, Y2, name='Vector_Y3D')

cross_product = vector_3D.cross(vector_Y3D)
cross_product.name = 'Cross Product'
samples = cross_product.sample(size=5)
print(f"Cross Product Samples:\n{samples}")

# ---------------------------
# Section 5: Dependency Structures
# ---------------------------

print("\nSection 5: Dependency Structures")

# Create dependencies
Base = StochasticVariable(ExponentialDistribution(lambd=1), name='Base')
Dependent = Base + 5
Dependent.name = 'Dependent = Base + 5'

# Sample with shared context
context = {}
base_samples = Base.sample(size=LARGE_SAMPLE, context=context)
dependent_samples = Dependent.sample(size=LARGE_SAMPLE, context=context)

# Verify dependency
expected_samples = base_samples + 5
assert np.allclose(dependent_samples, expected_samples), "Dependency not maintained!"

# Plot Dependency Graph
plot_dependency_graph(Dependent, title="Dependency Graph Example")

# ---------------------------
# Section 6: Monte Carlo Simulations
# ---------------------------

print("\nSection 6: Monte Carlo Simulations")

# Define a model function
def model_function(x_samples, y_samples):
    return np.exp(-x_samples) * np.sin(y_samples)

# Perform Monte Carlo simulation
X = StochasticVariable(ContinuousUniformDistribution(a=0, b=2*np.pi), name='X')
Y = StochasticVariable(ContinuousUniformDistribution(a=0, b=np.pi), name='Y')
variables = [X, Y]

results = monte_carlo_simulate(model_function, variables, trials=LARGE_SAMPLE)
print(f"Simulation Mean Result: {np.mean(results):.4f}")

# Summarize Simulation
summary = summarize_simulation(results)
print("Simulation Summary:")
for key, value in summary.items():
    if isinstance(value, tuple):  # Handle tuples like confidence intervals
        formatted_value = f"({value[0]:.4f}, {value[1]:.4f})"
    else:  # Handle numeric values
        formatted_value = f"{value:.4f}"
    print(f"{key}: {formatted_value}")

# Plot Simulation Results
plot_simulation(results, bins=30, title="Monte Carlo Simulation Results")

# ---------------------------
# Section 7: Probability Estimation
# ---------------------------

print("\nSection 7: Probability Estimation")

# Estimate probability that X + Y > Z
Z = StochasticVariable(NormalDistribution(mu=2, sigma=1), name='Z')

def condition(x_samples, y_samples, z_samples):
    return (x_samples + y_samples) > z_samples

prob = probability(condition, X, Y, Z, size=LARGE_SAMPLE)
print(f"Estimated Probability that X + Y > Z: {prob:.4f}")

# ---------------------------
# Section 8: Advanced Distributions
# ---------------------------

print("\nSection 8: Advanced Distributions")

# Dirichlet Distribution
alpha = [1, 2, 3]
dirichlet_dist = DirichletDistribution(alpha=alpha)
Dirichlet_Var = StochasticVariable(distribution=dirichlet_dist, name='Dirichlet_Var')

# Sample and display
samples = Dirichlet_Var.sample(size=5)
print(f"Dirichlet Distribution Samples:\n{samples}")

# ---------------------------
# Section 9: Goodness of Fit
# ---------------------------

print("\nSection 9: Goodness of Fit")

# Fit a normal distribution to data
data = np.random.normal(loc=5, scale=2, size=LARGE_SAMPLE)
fitted_dist = NormalDistribution.fit(data)
print(f"Fitted Parameters: mu = {fitted_dist.params['loc']:.4f}, sigma = {fitted_dist.params['scale']:.4f}")


# ---------------------------
# Section 10: Hypothesis Testing
# ---------------------------

print("\nSection 10: Hypothesis Testing")

# Test if the mean of a normal distribution is equal to a specific value
sample_mean = np.mean(data)
population_mean = 5
standard_error = np.std(data, ddof=1) / np.sqrt(len(data))
t_statistic = (sample_mean - population_mean) / standard_error
print(f"T-statistic: {t_statistic:.4f}")

# ---------------------------
# Section 11: Custom Distributions
# ---------------------------

print("\nSection 11: Custom Distributions")

# Define a custom PDF
def custom_pdf(x):
    return 3 * x**2  # PDF for x in [0, 1]

custom_dist = CustomDistribution(func=custom_pdf, domain=(0, 1), distribution_type='continuous')
Custom_Var = StochasticVariable(distribution=custom_dist, name='Custom_Var')

# Sample and plot
samples = Custom_Var.sample(size=LARGE_SAMPLE)
print(f"Custom Distribution Mean: {np.mean(samples):.4f}")
plot_distribution(Custom_Var, num_samples=1000, bins=30, title='Custom Distribution')

# ---------------------------
# Section 12: Mixture Distributions
# ---------------------------

print("\nSection 12: Mixture Distributions")

# Define mixture components
component1 = NormalDistribution(mu=-2, sigma=1)
component2 = NormalDistribution(mu=2, sigma=1)
weights = [0.3, 0.7]
mixture_dist = MixtureDistribution(components=[component1, component2], weights=weights)
Mixture_Var = StochasticVariable(distribution=mixture_dist, name='Mixture_Var')

# Sample and plot
samples = Mixture_Var.sample(size=LARGE_SAMPLE)
print(f"Mixture Distribution Mean: {np.mean(samples):.4f}")
plot_distribution(Mixture_Var, num_samples=1000, bins=30, title='Mixture Distribution')

# ---------------------------
# Section 13: Applying Constraints
# ---------------------------

print("\nSection 13: Applying Constraints")

# Sampling with constraints using rejection sampling
def positive_samples(variable, size):
    samples = []
    while len(samples) < size:
        sample = variable.sample(size=size)
        accepted = sample[sample > 0]
        samples.extend(accepted)
    return np.array(samples[:size])

X_Positive = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='X_Positive')
samples = positive_samples(X_Positive, LARGE_SAMPLE)
print(f"Mean of Positive Samples: {np.mean(samples):.4f}")

# ---------------------------
# Section 14: Law of Large Numbers
# ---------------------------

print("\nSection 14: Law of Large Numbers")

# Show convergence of sample mean to expected value
sample_sizes = [100, 1000, 10000, 100000]
expected_value = 0
for size in sample_sizes:
    samples = X.sample(size=size)
    sample_mean = np.mean(samples)
    print(f"Sample Size: {size}, Sample Mean: {sample_mean:.4f}")

# ---------------------------
# Section 15: Central Limit Theorem
# ---------------------------

print("\nSection 15: Central Limit Theorem")

# Sum of independent random variables approaches normal distribution
n_variables = 30
variables = [StochasticVariable(ExponentialDistribution(lambd=1), name=f'X_{i}') for i in range(n_variables)]
sum_variable = sum(variables)
samples = sum_variable.sample(size=LARGE_SAMPLE)
standardized_samples = (samples - n_variables) / np.sqrt(n_variables)
print(f"Standardized Samples Mean: {np.mean(standardized_samples):.4f}, Std Dev: {np.std(standardized_samples):.4f}")

# Plot standardized samples
plt.hist(standardized_samples, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
plt.title('Central Limit Theorem Demonstration')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# ---------------------------
# Section 16: Option Pricing with Monte Carlo Simulation
# ---------------------------

StochasticVariable.delete_all_instances()

print("\nSection 16: Option Pricing with Monte Carlo Simulation")

# European Call Option Pricing
S0 = 100  # Initial stock price
K = 105   # Strike price
T = 1     # Time to maturity
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

def option_payoff(Z_samples):
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z_samples)
    return np.exp(-r * T) * np.maximum(ST - K, 0)


Z = StochasticVariable(NormalDistribution(mu=0, sigma=1), name='Z')
results = monte_carlo_simulate(option_payoff, [Z], trials=LARGE_SAMPLE)
option_price = np.mean(results)
print(f"Estimated Option Price: {option_price:.4f}")

# Compare with Black-Scholes Formula
from scipy.stats import norm
d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
print(f"Black-Scholes Option Price: {bs_price:.4f}")


# ---------------------------
# Section 18: Estimating Integrals with Monte Carlo
# ---------------------------

print("\nSection 18: Estimating Integrals with Monte Carlo")

# Estimate integral of sin(x) from 0 to pi
def integrand(x_samples):
    return np.sin(x_samples)

X_uniform = StochasticVariable(ContinuousUniformDistribution(a=0, b=np.pi), name='X_uniform')
results = monte_carlo_simulate(integrand, [X_uniform], trials=LARGE_SAMPLE)
estimated_integral = (np.pi - 0) * np.mean(results)
actual_integral = 2  # Known value
print(f"Estimated Integral: {estimated_integral:.4f}, Actual Integral: {actual_integral}")

# ---------------------------
# Section 20: Applying Statistical Tests
# ---------------------------

print("\nSection 20: Applying Statistical Tests")

# Kolmogorov-Smirnov Test
from scipy.stats import kstest

# Test if samples from X follow a normal distribution
samples = X.sample(size=LARGE_SAMPLE)
ks_statistic, p_value = kstest(samples, 'norm', args=(mu, sigma))
print(f"KS Statistic: {ks_statistic:.4f}, p-value: {p_value:.4f}")

# ---------------------------
# End of Examples
# ---------------------------

print("\nEnd of Examples")
