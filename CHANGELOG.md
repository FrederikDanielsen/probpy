# Changelog

## [0.2.0] - 25-11-2024

### Added

- **Core Classes and Functions:**

  - **`StochasticVariable` Class:**
    - Represents random variables with support for:
      - Sampling from distributions.
      - Arithmetic operations with other stochastic variables and scalars (supports addition, subtraction, multiplication, division, and power).
      - Statistical methods:
        - `mean()`
        - `std()`
        - `var()`
        - `median()`
        - `mode()`
        - `nth_moment()`
        - `confidence_interval()`
      - Distribution methods:
        - `pdf()`
        - `pmf()`
        - `cdf()`
      - Empirical methods:
        - `empirical_pdf()`
        - `empirical_pmf()`
        - `empirical_cdf()`
      - Conditioning variables using the `given()` method.
      - Plotting distributions using the `plot()` method.
      - Circular dependency detection to prevent invalid variable definitions.
      - Handling constants and dynamic parameters in distributions.

  - **`StochasticVector` Class:**
    - Represents vectors of stochastic variables with support for:
      - Sampling vectors of variables.
      - Vector operations:
        - `norm(p=2)`: Computes the p-norm of the vector.
        - `dot(other)`: Computes the dot product with another vector.
        - `cross(other)`: Computes the cross product with another 3D vector.
      - Element-wise operations with overloaded operators for addition, subtraction, multiplication, and division.
      - Handling dependencies across all components.

  - **Core Functions:**
    - `apply(func, *args, name=None)`: Creates a new `StochasticVariable` by applying a function to existing variables or constants.
    - `probability(condition, *args, size=DEFAULT_STATISTICS_SAMPLE_SIZE)`: Estimates the probability of a condition involving stochastic variables.

- **Distributions:**

  - **Base Classes:**
    - `Distribution`: Base class for all distributions.
    - `ScipyDistribution`: Wrapper for SciPy's distributions, handling both discrete and continuous cases.
    - `CustomDistribution`: Allows user-defined distributions based on a lambda function.
    - `MixtureDistribution`: Represents a mixture of multiple distributions.

  - **Supported Discrete Distributions:**
    - `BernoulliDistribution`
    - `BinomialDistribution`
    - `GeometricDistribution`
    - `HypergeometricDistribution`
    - `PoissonDistribution`
    - `NegativeBinomialDistribution`
    - `MultinomialDistribution`
    - `DiscreteUniformDistribution`

  - **Supported Continuous Distributions:**
    - `NormalDistribution` (with `fit()` and `goodness_of_fit()` methods)
    - `ExponentialDistribution`
    - `GammaDistribution`
    - `BetaDistribution`
    - `ChiSquaredDistribution`
    - `RayleighDistribution`
    - `CauchyDistribution`
    - `DirichletDistribution`
    - `ContinuousUniformDistribution`
    - `StandardArcsineDistribution`

- **Transformations:**

  - Mathematical functions for transforming stochastic variables:
    - Exponential and logarithmic functions: `exp()`, `expm1()`, `log()`, `log10()`, `log2()`, `log1p()`.
    - Power functions: `sqrt()`, `square()`, `power()`, `cbrt()`, `reciprocal()`.
    - Trigonometric functions: `sin()`, `cos()`, `tan()`, `arcsin()`, `arccos()`, `arctan()`, `arctan2()`, `hypot()`.
    - Hyperbolic functions: `sinh()`, `cosh()`, `tanh()`, `arcsinh()`, `arccosh()`, `arctanh()`.
    - Rounding and clipping functions: `round()`, `floor()`, `ceil()`, `trunc()`, `clip()`.
    - Sign and comparison functions: `abs()`, `sign()`, `min()`, `max()`.

- **Monte Carlo Simulation:**

  - Functions for performing and analyzing simulations:
    - `monte_carlo_simulate(model, variables, trials, seed)`: Runs simulations of a model with stochastic inputs.
    - `summarize_simulation(results, confidence_level)`: Provides statistical summaries of simulation results.
    - `plot_simulation(results, bins, density, title)`: Visualizes simulation outputs.

- **Visualization Utilities:**

  - **Plotting Functions:**
    - `plot_distribution(stochastic_var, num_samples, bins, density, title)`: Plots the distribution of a stochastic variable.
    - `plot_dependency_graph(variables, title)`: Visualizes dependencies among stochastic variables, highlighting any circular dependencies.

- **Goodness of Fit Testing:**

  - **`goodness_of_fit(data, test)` Method:**
    - Available in `NormalDistribution` for statistical tests like Kolmogorov-Smirnov (`'ks'`), Chi-Squared (`'chi2'`), and Anderson-Darling (`'anderson'`).
    - Allows assessment of how well a distribution fits observed data.

- **Testing Suite:**

  - Comprehensive tests using Python's `unittest` framework:
    - Tests for distributions to verify correct sampling and statistical properties.
    - Tests for `StochasticVariable` operations, transformations, and methods.
    - Tests for vector operations in `StochasticVector`.
    - Tests for transformations to ensure mathematical functions behave correctly.
    - Tests for Monte Carlo simulations to verify accurate modeling and probability estimation.
    - Tests for plotting functions to ensure they execute without errors.

### Fixed

- **`apply()` Function:**
  - Corrected handling of constants and sample arrays to ensure accurate function application in stochastic computations.

- **`StochasticVector` Methods:**
  - Fixed the `norm()` method to compute per-sample norms correctly by stacking samples and specifying the correct axis.
  - Adjusted the `dot()` method to handle sample arrays properly for per-sample dot product calculations.
  - Fixed the `cross()` method to correctly capture loop variables and include all necessary dependencies.

- **`CustomDistribution` Sampling:**
  - Ensured that samples generated from `CustomDistribution` fall within the specified domain by implementing resampling and filtering.

- **Circular Dependency Detection:**
  - Improved detection and handling of circular dependencies in `StochasticVariable` to prevent infinite recursion and invalid variable definitions.

- **Test Corrections:**
  - Updated tests to reflect accurate statistical expectations and mathematical properties.
  - Corrected assumptions in tests regarding expected values, especially for operations involving random variables.

### Changed

- **`goodness_of_fit()` Method Usage:**
  - Restricted to appropriate distributions (e.g., `NormalDistribution`) where statistical tests are applicable and meaningful.

- **`apply()` Function Enhancements:**
  - Improved to handle constants more effectively by wrapping them as `StochasticVariable` instances with fixed values.
  - Ensured that functions receive correctly shaped sample arrays for accurate computations.

- **Documentation and Code Comments:**
  - Added detailed docstrings and comments to enhance code readability and maintainability.
  - Clarified explanations of functions, methods, and parameters.

- **Plotting Enhancements:**
  - Improved plotting functions to provide clearer visualizations.
  - Added handling for edge cases and ensured compatibility with various data types.
