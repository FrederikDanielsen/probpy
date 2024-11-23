import matplotlib.pyplot as plt
import numpy as np
from .core import DEFAULT_SAMPLE_SIZE


def plot_distribution(stochastic_var, num_samples=DEFAULT_SAMPLE_SIZE, bins=30, density=True, title=None):
    """
    Plots the distribution of a StochasticVariable and includes a vertical line for the mean.

    Parameters:
        stochastic_var (StochasticVariable): The stochastic variable to be plotted.
        num_samples (int): Number of samples to draw for the plot (default: 1000).
        bins (int): Number of bins in the histogram (default: 30).
        density (bool): Whether to normalize the histogram to show density (default: True).
        title (str): Title for the plot (optional).
    """
    # Generate samples
    samples = stochastic_var.sample(size=num_samples)
    mean_value = np.mean(samples)  # Calculate the mean of the samples

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=bins, density=density, alpha=0.6, color='blue', edgecolor='black', label='Histogram')

    # Add a density line if it's a continuous variable
    if density:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(samples)
        x_range = np.linspace(min(samples), max(samples), 1000)
        plt.plot(x_range, kde(x_range), color='red', label='Density')

    # Add a vertical line for the mean
    plt.axvline(mean_value, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.2f}')

    # Set labels and title
    plt.xlabel('Value')
    plt.ylabel('Density' if density else 'Frequency')
    if title is None:
        title = f'Distribution of {stochastic_var.name}'
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
