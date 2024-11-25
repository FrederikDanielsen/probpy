# goodness_of_fit.py

from scipy.stats import kstest, chi2, anderson
import numpy as np


def goodness_of_fit(self, data, test="ks"):
    """
    Performs a goodness-of-fit test.

    Parameters:
    - data (array-like): Observed data to compare against the distribution.
    - test (str): The goodness-of-fit test to perform ('ks', 'chi2', 'anderson').

    Returns:
    - dict: Results of the goodness-of-fit test.
    """
    if test == "ks":
        # Kolmogorov-Smirnov Test
        if self.distribution_type != "continuous":
            raise ValueError("KS test is only suitable for continuous distributions.")
        if not hasattr(self, "cdf"):
            raise NotImplementedError("KS test requires the distribution to have a cdf method.")
        
        # Use the built-in CDF for the test
        stat, p_value = kstest(data, self.cdf)
        return {"test": "Kolmogorov-Smirnov", "statistic": stat, "p_value": p_value}

    elif test == "chi2":
        # Chi-Square Test
        if self.distribution_type != "discrete":
            raise ValueError("Chi-Square test is suitable for discrete distributions or binned data.")
        if not hasattr(self, "pmf"):
            raise NotImplementedError("Chi-Square test requires the distribution to have a pmf method.")
        
        # Calculate observed and expected frequencies
        observed, bins = np.histogram(data, bins="auto")
        expected = len(data) * np.array([self.pmf((bins[i] + bins[i+1]) / 2) for i in range(len(bins) - 1)])
        stat = np.sum((observed - expected)**2 / expected)
        dof = len(observed) - 1
        p_value = 1 - chi2.cdf(stat, df=dof)
        return {"test": "Chi-Square", "statistic": stat, "p_value": p_value, "degrees_of_freedom": dof}

    elif test == "anderson":
        # Anderson-Darling Test
        if self.distribution_type != "continuous":
            raise ValueError("Anderson-Darling test is only suitable for continuous distributions.")
        if not hasattr(self, "cdf"):
            raise NotImplementedError("Anderson-Darling test requires the distribution to have a cdf method.")
        
        # Scipy's Anderson-Darling implementation is limited to normal by default.
        result = anderson(data, dist="norm")
        return {"test": "Anderson-Darling", "statistic": result.statistic, "critical_values": result.critical_values}

    else:
        raise ValueError("Unsupported test type. Use 'ks', 'chi2', or 'anderson'.")
