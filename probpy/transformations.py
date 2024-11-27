#transformations.py

# IMPORTS
from .core import StochasticVariable, apply
import numpy as np

# Exponential and logarithmic functions
def exp(X):
    return apply(np.exp, X, name=f"exp({X.name if isinstance(X, StochasticVariable) else X})")

def expm1(X):
    return apply(np.expm1, X, name=f"expm1({X.name if isinstance(X, StochasticVariable) else X})")

def log(X):
    return apply(np.log, X, name=f"log({X.name if isinstance(X, StochasticVariable) else X})")

def log10(X):
    return apply(np.log10, X, name=f"log10({X.name if isinstance(X, StochasticVariable) else X})")

def log2(X):
    return apply(np.log2, X, name=f"log2({X.name if isinstance(X, StochasticVariable) else X})")

def log1p(X):
    return apply(np.log1p, X, name=f"log1p({X.name if isinstance(X, StochasticVariable) else X})")

# Power functions
def sqrt(X):
    return apply(np.sqrt, X, name=f"sqrt({X.name if isinstance(X, StochasticVariable) else X})")

def square(X):
    return apply(np.square, X, name=f"square({X.name if isinstance(X, StochasticVariable) else X})")

def power(X, y):
    return apply(np.power, X, y, name=f"power({X.name if isinstance(X, StochasticVariable) else X}, {y.name if isinstance(y, StochasticVariable) else y})")

def cbrt(X):
    return apply(np.cbrt, X, name=f"cbrt({X.name if isinstance(X, StochasticVariable) else X})")

def reciprocal(X):
    return apply(np.reciprocal, X, name=f"reciprocal({X.name if isinstance(X, StochasticVariable) else X})")

# Trigonometric functions
def sin(X):
    return apply(np.sin, X, name=f"sin({X.name if isinstance(X, StochasticVariable) else X})")

def cos(X):
    return apply(np.cos, X, name=f"cos({X.name if isinstance(X, StochasticVariable) else X})")

def tan(X):
    return apply(np.tan, X, name=f"tan({X.name if isinstance(X, StochasticVariable) else X})")

def arcsin(X):
    return apply(np.arcsin, X, name=f"arcsin({X.name if isinstance(X, StochasticVariable) else X})")

def arccos(X):
    return apply(np.arccos, X, name=f"arccos({X.name if isinstance(X, StochasticVariable) else X})")

def arctan(X):
    return apply(np.arctan, X, name=f"arctan({X.name if isinstance(X, StochasticVariable) else X})")

def arctan2(X, Y):
    return apply(np.arctan2, X, Y, name=f"arctan2({X.name if isinstance(X, StochasticVariable) else X}, {Y.name if isinstance(Y, StochasticVariable) else Y})")

def hypot(X, Y):
    return apply(np.hypot, X, Y, name=f"hypot({X.name if isinstance(X, StochasticVariable) else X}, {Y.name if isinstance(Y, StochasticVariable) else Y})")

# Hyperbolic functions
def sinh(X):
    return apply(np.sinh, X, name=f"sinh({X.name if isinstance(X, StochasticVariable) else X})")

def cosh(X):
    return apply(np.cosh, X, name=f"cosh({X.name if isinstance(X, StochasticVariable) else X})")

def tanh(X):
    return apply(np.tanh, X, name=f"tanh({X.name if isinstance(X, StochasticVariable) else X})")

def arcsinh(X):
    return apply(np.arcsinh, X, name=f"arcsinh({X.name if isinstance(X, StochasticVariable) else X})")

def arccosh(X):
    return apply(np.arccosh, X, name=f"arccosh({X.name if isinstance(X, StochasticVariable) else X})")

def arctanh(X):
    return apply(np.arctanh, X, name=f"arctanh({X.name if isinstance(X, StochasticVariable) else X})")

# Round and clipping functions
def round_(X, decimals=0):  # Avoid overriding built-in 'round'
    def func(x):
        return np.round(x, decimals=decimals)
    return apply(func, X, name=f"round({X.name if isinstance(X, StochasticVariable) else X}, decimals={decimals})")

def floor(X):
    return apply(np.floor, X, name=f"floor({X.name if isinstance(X, StochasticVariable) else X})")

def ceil(X):
    return apply(np.ceil, X, name=f"ceil({X.name if isinstance(X, StochasticVariable) else X})")

def trunc(X):
    return apply(np.trunc, X, name=f"trunc({X.name if isinstance(X, StochasticVariable) else X})")

def clip(X, a_min, a_max):
    def func(x):
        return np.clip(x, a_min, a_max)
    return apply(func, X, name=f"clip({X.name if isinstance(X, StochasticVariable) else X}, {a_min}, {a_max})")

# Sign and comparison functions
def abs_(X):  # Avoid overriding built-in 'abs'
    return apply(np.abs, X, name=f"abs({X.name if isinstance(X, StochasticVariable) else X})")

def sign(X):
    return apply(np.sign, X, name=f"sign({X.name if isinstance(X, StochasticVariable) else X})")

def min_(*Xs):  # Avoid overriding built-in 'min'
    names = ', '.join([x.name if isinstance(x, StochasticVariable) else str(x) for x in Xs])
    return apply(lambda *args: np.minimum.reduce(args), *Xs, name=f"min({names})")

def max_(*Xs):  # Avoid overriding built-in 'max'
    names = ', '.join([x.name if isinstance(x, StochasticVariable) else str(x) for x in Xs])
    return apply(lambda *args: np.maximum.reduce(args), *Xs, name=f"max({names})")
