from probpy import probability as P
from probpy.distributions import (
    StochasticVariable,
    DiscreteUniformDistribution,
    PoissonDistribution,
    NormalDistribution,
)


def main():
    # Example 1: Creating Stochastic Variables
    print("Example 1: Creating Stochastic Variables")
    X = StochasticVariable(DiscreteUniformDistribution(a=1, b=10), name="Uniform(1, 10)")
    Y = StochasticVariable(PoissonDistribution(lambda_=5), name="Poisson(λ=5)")
    print(f"Samples from X (Uniform(1, 10)): {X.sample(size=5)}")
    print(f"Samples from Y (Poisson(λ=5)): {Y.sample(size=5)}")
    print()

    # Example 2: Combining Stochastic Variables
    print("Example 2: Combining Stochastic Variables")
    Z = X + Y
    print(f"Samples from Z (X + Y): {Z.sample(size=5)}")
    print(f"Mean of Z: {Z.mean(size=1000)}")
    print(f"Variance of Z: {Z.std(size=1000)**2}")
    print()

    # Example 3: Complex Expressions with Stochastic Variables
    print("Example 3: Complex Expressions with Stochastic Variables")
    Z = X * 3 - Y / 2 + 6.5
    print(f"Samples from Z (X * 3 - Y / 2 + 6.5): {Z.sample(size=5)}")
    print(f"Mean of Z: {Z.mean(size=1000)}")
    print(f"Variance of Z: {Z.std(size=1000)**2}")
    print()

    # Example 4: Stochastic Variable as a Parameter
    print("Example 4: Stochastic Variable as a Parameter")
    lambda_var = StochasticVariable(DiscreteUniformDistribution(a=1, b=10), name="Lambda (Uniform)")
    Z = StochasticVariable(PoissonDistribution(lambda_=lambda_var), name="Poisson with Stochastic λ")
    print(f"Samples from Z (Poisson with Stochastic λ): {Z.sample(size=5)}")
    print(f"Mean of Z: {Z.mean(size=1000)}")
    print()

    # Example 5: Confidence Intervals and Higher Moments
    print("Example 5: Confidence Intervals and Higher Moments")
    X = StochasticVariable(NormalDistribution(mu=0, sigma=1), name="Normal(0, 1)")
    Y = StochasticVariable(NormalDistribution(mu=2, sigma=3), name="Normal(2, 3)")

    Z = 2* X + Y
    ci = Z.confidence_interval(confidence_level=0.95, size=1000)
    third_moment = Z.moment(n=3, size=1000)

    print(f"95% Confidence Interval for Z: {ci}")
    print(f"Third Moment of Z: {third_moment}")
    print()

    # Example 6: Empirical Probability
    print("Example 6: Empirical Probability")
    probability = P(
        X, Y,
        condition=lambda x, y: x + y > 0,
        size=1000
    )
    print(f"Empirical Probability that X + Y > 0: {probability}")
    print()


if __name__ == "__main__":
    main()
