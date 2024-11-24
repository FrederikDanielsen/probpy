from probpy.distributions import StochasticVariable, NormalDistribution, ContinuousUniformDistribution
import probpy as pp
from probpy import probability as P
from probpy.plots import plot_distribution

if __name__ == "__main__":
    X = StochasticVariable(NormalDistribution(0, 1))
    Y = X**2 + 3*X

    print("P(Y < 2) =", P(Y, condition=lambda y: y < 2))
    

    Z = X + X



