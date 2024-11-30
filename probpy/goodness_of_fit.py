# goodness_of_fit.py

# Imports
from .distributions import Distribution
from .core import StochasticVariable
from .constants import DEFAULT_STATISTICS_SAMPLE_SIZE
from scipy.stats import kstest, chisquare
import numpy as np


def kolmogorov_smirnov_test(subject, X, summary=True, alpha=0.05):
    """
    Description: Performs a Komogorov-Smirnov goodness of fit test on data or a StochasticVariable.

    Parameters:
        - subject (Distribution or StochasticVariable): supplies the distribution
        - X (StochasticVariable or list): supplies the data. If StochasticVariable the function samples automatically. Can, however, also be a list of numbers.
    
    returns: 
        - ks_stat (float): The Komogorov-Smirnov test statistic.
        - p_value (float): the p_value as a pair.

    """

    if isinstance(subject, StochasticVariable):
        dist = subject.distribution
    else:
        dist = subject

    if isinstance(X, StochasticVariable):
        data = X.sample(size=DEFAULT_STATISTICS_SAMPLE_SIZE)

    ks_stat, p_value = kstest(data, dist.cdf)
    
    if summary:
        print("\n--------------------------------------------------------------------")
        print("                         Kolmogorov-Smirnov test")
        print("--------------------------------------------------------------------")
        print(f"KS Statistic: {ks_stat}")
        print(f"P-value: {p_value}\n")

        # Interpretation
        if p_value > alpha:
            print(f"The {"StochasticVariable " + X.name if isinstance(X, StochasticVariable) else "data"} fits the distribution. \nCONCLUSION: fail to reject the null hypothesis.")
        else:
            print(f"The {"StochasticVariable " + X.name if isinstance(X, StochasticVariable) else "data"} does not fit the distribution. \nCONCLUSION: reject the null hypothesis.")

        print("--------------------------------------------------------------------")
        print("\nNote: The p-value represents the probability of observing results as \nextreme as the current data, assuming the null hypothesis is true.\nA low p-value (e.g., ≤ 0.05) indicates strong evidence against the \nnull hypothesis, while a high p-value suggests the data is \nconsistent with the null hypothesis.\n")

    return ks_stat, p_value


def chi_square_test(subject, X, summary=True, alpha=0.05):
    """
    Description: Performs a Chi-square goodness of fit test on data or a StochasticVariable.

    Parameters:
        - subject (Distribution or StochasticVariable): supplies the distribution
        - X (StochasticVariable or list): supplies the data. If StochasticVariable the function samples automatically. Can, however, also be a list of numbers.
    
    returns: 
        - ks_stat (float): The Chi-square test statistic.
        - p_value (float): the p_value as a pair.

    """


    if isinstance(subject, StochasticVariable):
        dist = subject.distribution
    else:
        dist = subject

    if isinstance(X, StochasticVariable):
        data = X.sample(size=DEFAULT_STATISTICS_SAMPLE_SIZE)

    # Compute bins and observed frequencies
    bins = np.histogram_bin_edges(data, bins='auto')
    observed, _ = np.histogram(data, bins=bins)

    # Compute expected frequencies
    expected = len(data) * (dist.cdf(bins[1:]) - dist.cdf(bins[:-1]))

    # Normalize expected to match observed total
    expected *= observed.sum() / expected.sum()

    # Perform chi-square test
    chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    if summary:
        print("\n--------------------------------------------------------------------")
        print("                          Chi-square test")
        print("--------------------------------------------------------------------")
        print(f"Chi-Square Statistic: {chi_stat}")
        print(f"P-value: {p_value}\n")

        # Interpret the result
        if p_value > alpha:
            print(f"The {"StochasticVariable " + X.name if isinstance(X, StochasticVariable) else "data"} fits the distribution \nCONCLUSION: fail to reject the null hypothesis.")
        else:
            print(f"The {"StochasticVariable " + X.name if isinstance(X, StochasticVariable) else "data"} does not fit the distribution \nCONCLUSION: reject the null hypothesis.")

        print("--------------------------------------------------------------------")
        print("\nNote: The p-value represents the probability of observing results as \nextreme as the current data, assuming the null hypothesis is true.\nA low p-value (e.g., ≤ 0.05) indicates strong evidence against the \nnull hypothesis, while a high p-value suggests the data is \nconsistent with the null hypothesis.\n")


    return chi_stat, p_value