# mmonte_carlo.py

import numpy as np
from .core import StochasticVariable


def monte_carlo_simulate(model, variables: list[StochasticVariable], trials=10000, seed=None):
    """
    Performs Monte Carlo simulation.

    Parameters:
        - model (callable): A function that takes samples from the variables and returns a result.
        - variables (list of StochasticVariable): Input stochastic variables for the model.
        - trials (int): Number of Monte Carlo trials (default: 10000).
        - seed (int): Random seed for reproducibility (default: None).

    Returns:
        - results (numpy.ndarray): Array of simulation results.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate samples for each stochastic variable
    samples = [var.sample(size=trials) for var in variables]

    # Evaluate the model for each set of sampled inputs
    results = np.array([model(*args) for args in zip(*samples)])

    return results


def summarize_simulation(results, confidence_level=0.95):
    """
    Summarizes Monte Carlo simulation results with basic statistics.

    Parameters:
        - results (numpy.ndarray): Simulation results.
        - confidence_level (float): Confidence level for the interval (default: 0.95).

    Returns:
        - summary (dict): A dictionary containing mean, variance, standard deviation, median,
                          and confidence interval.
    """
    mean = np.mean(results)
    variance = np.var(results)
    std_dev = np.std(results)
    median = np.median(results)

    # Confidence interval
    alpha = 1 - confidence_level
    lower_bound = np.percentile(results, 100 * alpha / 2)
    upper_bound = np.percentile(results, 100 * (1 - alpha / 2))

    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "median": median,
        "confidence_interval": (lower_bound, upper_bound),
    }


def plot_simulation(results, bins=30, density=True, title="Monte Carlo Simulation Results"):
    """
    Plots the distribution of simulation results.

    Parameters:
        - results (numpy.ndarray): Simulation results.
        - bins (int): Number of bins for the histogram (default: 30).
        - density (bool): Whether to normalize the histogram (default: True).
        - title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    plt.figure(figsize=(8, 6))

    # Plot histogram
    plt.hist(results, bins=bins, density=density, alpha=0.6, color='blue', edgecolor='black', label='Histogram')

    # Add density line
    if density:
        kde = gaussian_kde(results)
        x_range = np.linspace(min(results), max(results), 1000)
        plt.plot(x_range, kde(x_range), color='red', label='KDE')

    # Add mean line
    mean = np.mean(results)
    plt.axvline(mean, color='green', linestyle='--', linewidth=2, label=f"Mean = {mean:.2f}")

    # Final touches
    plt.xlabel('Simulation Output')
    plt.ylabel('Density' if density else 'Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
