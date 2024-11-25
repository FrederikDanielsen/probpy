#transformations.py

import numpy as np
from .core import apply


# Exponential and logarithmic functions
def exp(X):
    return apply(np.exp, X, name=f"exp({X.name})" if hasattr(X, "name") else "exp")

def expm1(X):
    return apply(np.expm1, X, name=f"expm1({X.name})" if hasattr(X, "name") else "expm1")

def log(X):
    return apply(np.log, X, name=f"log({X.name})" if hasattr(X, "name") else "log")

def log10(X):
    return apply(np.log10, X, name=f"log10({X.name})" if hasattr(X, "name") else "log10")

def log2(X):
    return apply(np.log2, X, name=f"log2({X.name})" if hasattr(X, "name") else "log2")

def log1p(X):
    return apply(np.log1p, X, name=f"log1p({X.name})" if hasattr(X, "name") else "log1p")


# Power functions
def sqrt(X):
    return apply(np.sqrt, X, name=f"sqrt({X.name})" if hasattr(X, "name") else "sqrt")

def square(X):
    return apply(np.square, X, name=f"square({X.name})" if hasattr(X, "name") else "square")

def power(X, y):
    x_name = X.name if hasattr(X, "name") else "X"
    y_name = y.name if hasattr(y, "name") else str(y)
    return apply(np.power, X, y, name=f"power({x_name}, {y_name})")

def cbrt(X):
    return apply(np.cbrt, X, name=f"cbrt({X.name})" if hasattr(X, "name") else "cbrt")

def reciprocal(X):
    return apply(lambda x: 1 / x, X, name=f"reciprocal({X.name})" if hasattr(X, "name") else "reciprocal")


# Trigonometric functions
def sin(X):
    return apply(np.sin, X, name=f"sin({X.name})" if hasattr(X, "name") else "sin")

def cos(X):
    return apply(np.cos, X, name=f"cos({X.name})" if hasattr(X, "name") else "cos")

def tan(X):
    return apply(np.tan, X, name=f"tan({X.name})" if hasattr(X, "name") else "tan")

def arcsin(X):
    return apply(np.arcsin, X, name=f"arcsin({X.name})" if hasattr(X, "name") else "arcsin")

def arccos(X):
    return apply(np.arccos, X, name=f"arccos({X.name})" if hasattr(X, "name") else "arccos")

def arctan(X):
    return apply(np.arctan, X, name=f"arctan({X.name})" if hasattr(X, "name") else "arctan")

def arctan2(X, Y):
    x_name = X.name if hasattr(X, "name") else "X"
    y_name = Y.name if hasattr(Y, "name") else "Y"
    return apply(np.arctan2, X, Y, name=f"arctan2({x_name}, {y_name})")

def hypot(X, Y):
    x_name = X.name if hasattr(X, "name") else "X"
    y_name = Y.name if hasattr(Y, "name") else "Y"
    return apply(np.hypot, X, Y, name=f"hypot({x_name}, {y_name})")


# Hyperbolic functions
def sinh(X):
    return apply(np.sinh, X, name=f"sinh({X.name})" if hasattr(X, "name") else "sinh")

def cosh(X):
    return apply(np.cosh, X, name=f"cosh({X.name})" if hasattr(X, "name") else "cosh")

def tanh(X):
    return apply(np.tanh, X, name=f"tanh({X.name})" if hasattr(X, "name") else "tanh")

def arcsinh(X):
    return apply(np.arcsinh, X, name=f"arcsinh({X.name})" if hasattr(X, "name") else "arcsinh")

def arccosh(X):
    return apply(np.arccosh, X, name=f"arccosh({X.name})" if hasattr(X, "name") else "arccosh")

def arctanh(X):
    return apply(np.arctanh, X, name=f"arctanh({X.name})" if hasattr(X, "name") else "arctanh")


# Round and clipping
def round(X, decimals=0):
    return apply(lambda x: np.round(x, decimals=decimals), X, name=f"round({X.name}, {decimals})" if hasattr(X, "name") else f"round_{decimals}")

def floor(X):
    return apply(np.floor, X, name=f"floor({X.name})" if hasattr(X, "name") else "floor")

def ceil(X):
    return apply(np.ceil, X, name=f"ceil({X.name})" if hasattr(X, "name") else "ceil")

def trunc(X):
    return apply(np.trunc, X, name=f"trunc({X.name})" if hasattr(X, "name") else "trunc")

def clip(X, a_min, a_max):
    return apply(lambda x: np.clip(x, a_min, a_max), X, name=f"clip({X.name}, {a_min}, {a_max})" if hasattr(X, "name") else f"clip_{a_min}_{a_max}")


# Sign and comparison
def abs(X):
    return apply(np.abs, X, name=f"abs({X.name})" if hasattr(X, "name") else "abs")

def sign(X):
    return apply(np.sign, X, name=f"sign({X.name})" if hasattr(X, "name") else "sign")

def min(*Xs):
    x_names = [x.name if hasattr(x, "name") else "X" for x in Xs]
    return apply(np.min, *Xs, name=f"min({', '.join(x_names)})")

def max(*Xs):
    x_names = [x.name if hasattr(x, "name") else "X" for x in Xs]
    return apply(np.max, *Xs, name=f"max({', '.join(x_names)})")


