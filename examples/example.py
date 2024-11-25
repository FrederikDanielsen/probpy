import numpy as np
import matplotlib.pyplot as plt
from probpy.core import StochasticVariable, StochasticVector
from probpy.distributions import NormalDistribution, ContinuousUniformDistribution, DiscreteUniformDistribution, MixtureDistribution
from probpy.transformations import *
from probpy.plots import plot_dependency_graph

# Define distributions
normal_dist = NormalDistribution(mu=0, sigma=1)  # Standard Normal
uniform_dist = ContinuousUniformDistribution(a=0, b=1)  # Uniform between 0 and 1
discrete_dist = DiscreteUniformDistribution(a=1, b=10)  # Discrete Uniform between 1 and 10

# Create stochastic variables
X = StochasticVariable(normal_dist, name="X")
Y = StochasticVariable(uniform_dist, name="Y")
Z = StochasticVariable(discrete_dist, name="Z")

# Sampling and plotting
samples_X = X.sample(size=1000)
samples_Y = Y.sample(size=1000)
samples_Z = Z.sample(size=1000)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(samples_X, bins=30, alpha=0.6, label="X (Normal)")
plt.title("X (Normal Distribution)")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(samples_Y, bins=30, alpha=0.6, label="Y (Uniform)")
plt.title("Y (Uniform Distribution)")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(samples_Z, bins=10, alpha=0.6, label="Z (Discrete Uniform)")
plt.title("Z (Discrete Uniform Distribution)")
plt.legend()

plt.tight_layout()
plt.show()

# Apply transformations using `apply`
exp_X = exp(X)  # Exponential of X
sqrt_Y = sqrt(Y)  # Square root of Y
log_Z = log1p(Z)  # Natural log (log(1+Z))

# Sample transformed variables
samples_exp_X = exp_X.sample(size=1000)
samples_sqrt_Y = sqrt_Y.sample(size=1000)
samples_log_Z = log_Z.sample(size=1000)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(samples_exp_X, bins=30, alpha=0.6, label="exp(X)")
plt.title("exp(X)")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(samples_sqrt_Y, bins=30, alpha=0.6, label="sqrt(Y)")
plt.title("sqrt(Y)")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(samples_log_Z, bins=30, alpha=0.6, label="log1p(Z)")
plt.title("log1p(Z)")
plt.legend()

plt.tight_layout()
plt.show()

# Create stochastic vectors
vector1 = StochasticVector(X, Y, Z, name="Vector1")
vector2 = StochasticVector(exp_X, sqrt_Y, log_Z, name="Vector2")

# Vector operations using element-wise operations and apply
vector_sum = vector1 + vector2  # Element-wise addition
vector_mul = vector1 * vector2  # Element-wise multiplication
vector_dot = vector1.dot(vector2)  # Dot product
vector_cross = vector1.cross(vector2)  # Cross product (valid for 3D vectors)

# Sampling results
samples_sum = vector_sum.sample(size=5)
samples_mul = vector_mul.sample(size=5)
samples_dot = vector_dot.sample(size=5)

print("Samples from vector addition (element-wise):", samples_sum)
print("Samples from vector multiplication (element-wise):", samples_mul)
print("Samples from vector dot product:", samples_dot)

# Norms and scalar operations
vector_norm = vector1.norm(p=2)  # L2 norm
samples_norm = vector_norm.sample(size=5)
print("Samples from vector norm:", samples_norm)

scaled_vector = vector1 * 2  # Scalar multiplication
samples_scaled = scaled_vector.sample(size=5)
print("Samples from scaled vector:", samples_scaled)

# Mixture distributions
mixture = MixtureDistribution(
    components=[normal_dist, uniform_dist],
    weights=[0.7, 0.3]
)
mixed_var = StochasticVariable(mixture, name="MixedVar")

# Sample and plot mixture distribution
samples_mixture = mixed_var.sample(size=1000)

plt.figure(figsize=(6, 4))
plt.hist(samples_mixture, bins=30, alpha=0.6, label="Mixture Distribution")
plt.title("Mixture Distribution")
plt.legend()
plt.show()

# Goodness of fit for normal distribution
ks_test_result = normal_dist.goodness_of_fit(samples_X, test="ks")
print("Kolmogorov-Smirnov Test for X:", ks_test_result)

anderson_test_result = normal_dist.goodness_of_fit(samples_X, test="anderson")
print("Anderson-Darling Test for X:", anderson_test_result)

print("Vector1 dependencies:", [dep.name for dep in vector1.get_all_dependencies()])
print("Vector2 dependencies:", [dep.name for dep in vector2.get_all_dependencies()])


# Dependency graph
plot_dependency_graph([X, Y, Z, exp_X, sqrt_Y, log_Z, vector1, vector2])
