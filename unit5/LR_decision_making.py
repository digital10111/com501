import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def log2(x):
    if not isinstance(x, np.ndarray): x = np.array(x)
    xx = x.copy()
    x[xx == 0] = 1
    return np.log2(xx)


def H_2(x, t):
    return -(t*log2(x) + (1-t)*log2(1-x))


def H2(x):
    return H_2(x, x)


