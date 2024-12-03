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


# TODO

# 1.  (DONE) Goodness of fit
# 2.  Conditional distributions (X.given(Y=y))
# 3.  Multivariate distributions
# 4.  (DONE) Stochastic matrices
# 5.  (DONE) Dont include constants in graphs
# 6.  (DONE) Add distribution of instantiating empty stochastiv variable.
# 7.  (DONE) As default be Contiuously distributed on the unit interval
# 8.  (DONE) Cleanup: Make the structure of the codebase more rational. Remove that func 
#     attribute of the StochasticVariable class and replace it by a CustomDistribution
# 9.  (DONE) Remove distribution_type attribute from StochasticVariable. Instead draw it
#     from the added distribution. 8. needs to be implemented first.
# 10. Include plotting of vector distributions. Maybe this is just plotting of multivariate distributions?
# 11. Include more methods in Matrix class - like matrix norm, maybe something like matrix exponential.
# 12. Make it possible to apply transformations elementwise to vectors and matrices