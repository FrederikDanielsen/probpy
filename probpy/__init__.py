# __init__.py

# Import key functions and classes for easy access
from .constants import DEFAULT_PLOTTING_SAMPLE_SIZE, DEFAULT_STATISTICS_SAMPLE_SIZE
from .core import apply, probability, StochasticVariable, StochasticVector
from .transformations import *

__version__ = "0.1.0"
__author__ = "Frederik Danielsen"
