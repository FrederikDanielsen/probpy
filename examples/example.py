from probpy import probability as P
import probpy as pp
from probpy.plots import plot_distribution
from probpy.distributions import StochasticVariable, NormalDistribution, ContinuousUniformDistribution
import numpy as np

X = StochasticVariable(ContinuousUniformDistribution(-2,3))
Y = StochasticVariable(NormalDistribution(X,1))

Z = pp.min(X, Y)



probability = P(Z, condition=lambda z: z < 0)


print("P(Z < 0) =", probability)


plot_distribution(Z)