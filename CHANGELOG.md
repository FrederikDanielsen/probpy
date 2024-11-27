# Changelog

## [0.3.0] - 26-11-2024

### Added

#### Core Classes and Functions:

- **`StochasticVariable` Class**:
  - Supports overloaded arithmetic operators for intuitive mathematical expressions involving stochastic variables.
  - Provides statistical methods:
    - `mean()`
    - `std()`
    - `var()`
    - `median()`
    - `mode()`
    - `nth_moment()`
    - `confidence_interval()`
  - Offers distribution methods:
    - `pdf()`
    - `pmf()`
    - `cdf()`
  - Includes empirical methods:
    - `empirical_pdf()`
    - `empirical_pmf()`
    - `empirical_cdf()`
  - Implements circular dependency detection to prevent invalid variable definitions.

- **`StochasticVector` Class**:
  - Supports vector operations:
    - `norm(p=2)`: Computes the p-norm of the vector.
    - `dot(other)`: Computes the dot product with another vector.
    - `cross(other)`: Computes the cross product with another 3D vector.
  - Provides overloaded operators for element-wise addition, subtraction, multiplication, and division.
  - Manages dependencies across all components.

#### Distributions:

- **`StandardDistribution` Class**:
  - Renamed from `ScipyDistribution` for clarity.
  - Wraps SciPy's standard distributions, handling both discrete and continuous cases.
- **`CustomDistribution` Class**:
  - Allows user-defined distributions based on custom functions.
  - Supports sampling and computing PDF/PMF over specified domains.

#### Monte Carlo Simulation:

- Functions for performing and analyzing simulations:
  - `monte_carlo_simulate(model, variables, trials, seed)`: Runs simulations of a model with stochastic inputs.
  - `summarize_simulation(results, confidence_level)`: Provides statistical summaries of simulation results.
  - `plot_simulation(results, bins, density, title)`: Visualizes simulation outputs.

#### Visualization Utilities:

- **Plotting Functions**:
  - `plot_distribution(stochastic_var, num_samples, bins, density, title)`: Plots the distribution of a stochastic variable.
  - `plot_dependency_graph(variables, title)`: Visualizes dependencies among stochastic variables, highlighting any circular dependencies.

#### Transformations:

- Mathematical functions for transforming stochastic variables:
  - **Exponential and logarithmic functions**: `exp()`, `expm1()`, `log()`, `log10()`, `log2()`, `log1p()`.
  - **Power functions**: `sqrt()`, `square()`, `power()`, `cbrt()`, `reciprocal()`.
  - **Trigonometric functions**: `sin()`, `cos()`, `tan()`, `arcsin()`, `arccos()`, `arctan()`, `arctan2()`, `hypot()`.
  - **Hyperbolic functions**: `sinh()`, `cosh()`, `tanh()`, `arcsinh()`, `arccosh()`, `arctanh()`.
  - **Rounding and clipping functions**: `round_()`, `floor()`, `ceil()`, `trunc()`, `clip()`.
  - **Sign and comparison functions**: `abs_()`, `sign()`, `min_()`, `max_()`.

---

### Removed

#### Methods:

- **`given()` Method**:
  - Removed from `StochasticVariable` to simplify the class interface.
  - Conditioning variables now requires alternative approaches.
- **`plot()` Method**:
  - Removed from `StochasticVariable` in favor of standalone plotting functions in the `plots` module.

#### Classes:

- **`MixtureDistribution` Class**:
  - Removed to streamline the distributions module.
  - Users can implement mixtures using custom distributions or combine samples manually.

#### Goodness of Fit Testing:

- **`goodness_of_fit()` Method**:
  - Removed from `NormalDistribution` due to limited applicability.
  - Users should utilize external statistical tests for goodness-of-fit assessments.

#### Testing Suite:

- The comprehensive testing suite has been removed from the library distribution.
- Internal tests are maintained separately to ensure code reliability.

---

### Fixed

- **Sampling in `CustomDistribution`**:
  - Improved sampling using rejection sampling to ensure generated samples fall within the specified domain.
  - Enhanced efficiency and accuracy in sample generation for custom distributions.

- **Circular Dependency Detection**:
  - Enhanced detection and prevention of circular dependencies in `StochasticVariable`.
  - Ensures valid variable definitions and prevents infinite recursion.

- **Vector Operations in `StochasticVector`**:
  - Fixed the `norm()`, `dot()`, and `cross()` methods to correctly handle sample arrays and per-sample computations.
  - Resolved issues with dependency tracking and sample alignment across vector components.

---

### Changed

#### Class Renaming:

- **`ScipyDistribution` Renamed to `StandardDistribution`**:
  - Provides a clearer description of the class's purpose.
  - Continues to wrap standard distributions from SciPy.

#### `apply()` Function Enhancements:

- Improved handling of constants by automatically wrapping them as `StochasticVariable` instances with fixed values.
- Ensured that functions receive correctly shaped sample arrays for accurate computations.
- Enhanced flexibility in combining stochastic variables with constants.

#### Removal of Goodness of Fit Testing:

- The `goodness_of_fit()` method has been removed from `NormalDistribution`.
- Encourages users to apply specialized statistical packages for rigorous goodness-of-fit testing.

#### Documentation and Code Comments:

- Added detailed docstrings and comments throughout the codebase.
- Improved explanations of functions, methods, and parameters for better developer understanding.

#### Plotting Enhancements:

- Refined plotting functions to provide clearer and more informative visualizations.
- Improved handling of edge cases and compatibility with various data types.

#### Simplification of `StochasticVariable`:

- Streamlined the class by removing less commonly used methods (`given()` and `plot()`).
- Focused on core functionalities and statistical computations.

---

This update reflects significant enhancements to the library, focusing on improving core functionalities, refining existing methods, and simplifying the interface for better usability. Users are encouraged to explore the new features and adjust their codebases accordingly to accommodate the changes.
