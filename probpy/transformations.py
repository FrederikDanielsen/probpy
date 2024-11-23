import numpy as np
from .core import apply

# Exponential and logarithmic functions

def exp(X):
    return apply(X, transformation=np.exp, name="exp")

def expm1(X):
    return apply(X, transformation=np.expm1, name="expm1")

def log(X):
    return apply(X, transformation=np.log, name="log")

def log10(X):
    return apply(X, transformation=np.log10, name="log10")

def log2(X):
    return apply(X, transformation=np.log2, name="log2")

def log1p(X):
    return apply(X, transformation=np.log1p, name="log1p")


# Power functions

def sqrt(X):
    return apply(X, transformation=np.sqrt, name="sqrt")

def square(X):
    return apply(X, transformation=np.square, name="square")

def power(X, y):
    return X ** y

def cbrt(X):
    return apply(X, transformation=np.cbrt, name="cbrt")

def reciprocal(X):
    return apply(X, transformation=np.reciprocal, name="reciprocal")


# Trigonometric functions

def sin(X):
    return apply(X, transformation=np.sin, name="sin")

def cos(X):
    return apply(X, transformation=np.cos, name="cos")

def tan(X):
    return apply(X, transformation=np.tan, name="tan")

def arcsin(X):
    return apply(X, transformation=np.arcsin, name="arcsin")

def arccos(X):
    return apply(X, transformation=np.arccos, name="arccos")

def arctan(X):
    return apply(X, transformation=np.arctan, name="arctan")

def arctan2(X, Y):
    return apply(X, Y, transformation=np.arctan, name="arctan")

def hypot(X, Y):
    return apply(X, Y, transformation=np.hypot, name="hypot")



# Hyperbolic functions

def sinh(X):
    return apply(X, transformation=np.sinh, name="sinh")

def cosh(X):
    return apply(X, transformation=np.cosh, name="cosh")

def tanh(X):
    return apply(X, transformation=np.tanh, name="tanh")

def arcsinh(X):
    return apply(X, transformation=np.arcsinh, name="arcsinh")

def arccosh(X):
    return apply(X, transformation=np.arccosh, name="arccosh")

def arctanh(X):
    return apply(X, transformation=np.arctanh, name="arctanh")


# Round and clipping

def round(X, decimals=1):
    return apply(X, transformation=lambda x: np.round(x, decimals), name="round")

def floor(X):
    return apply(X, transformation=np.floor, name="floor")

def ceil(X):
    return apply(X, transformation=np.ceil, name="ceil")

def trunc(X):
    return apply(X, transformation=np.trunc, name="trunc")

def clip(X, a_min, a_max):
    return apply(X, transformation=lambda x: np.trunc(x, a_min, a_max), name=f"clip_[{a_min}, {a_max}]")



# Sign and comparison

def abs(X):
    return apply(X, transformation=np.abs, name="abs")

def sign(X):
    return apply(X, transformation=np.sign, name="sign")

def min(*Xs):
    return apply(*Xs, transformation=lambda *x: np.min(x), name="sign")

def max(*Xs):
    return apply(*Xs, transformation=lambda *x: np.max(x), name="sign")

