# __init__.py

# Import key functions and classes for easy access
from .transformations import *
from .core import *
from .plots import *

__version__ = "0.1.0"
__author__ = "Frederik Danielsen"


# Explicitly define what gets imported when `from probpy import *` is used
__all__ = [
    # Add everything from core
    *core.__all__,  
    *plots.__all__,
    *transformations.__all__
]


# IDEAS FOR IMPROVEMENT
# - Goodness of fit
# - Conditional distributions (X.given(Y=y))
# - Multivariate distributions
# - Dont include constants in graphs
# - Stochastic matrices