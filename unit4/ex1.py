import numpy as np
from numpy.linalg import inv  # NumPy Linear algebra library
import matplotlib.pyplot as plt
import pandas as pd


def get_t(x, w_):
    w_ = np.array(w_)
    return w_.dot(x)


def get_data(w_, noise=0, n=1000):
    x1 = np.linspace(-10, 10, num=n)
    x2 = np.random.normal(x1, 5)
    x = np.array([np.ones((len(x1))), x1, x2])
    t = get_t(x, w_)
    t = np.random.normal(t, noise)
    return x, t


def get_LS(x, t):
    xt = x.dot(t)
    xx = inv(x.dot(x.T))
    xx = np.round(xx, 4)
    xt = np.round(xt, 2)
    w = xx.dot(xt)
    w = np.round(w, 2)

    return w


def applyLST(X, t):
    w = get_LS(X, t)  # solve using least squares
    y = get_t(X, w)  # predict using the solution

    RMSE = np.sqrt(((t - y) ** 2).sum() / len(t))  # get the Sqrt(Mean Sum of Squared Error)
    return y, w, RMSE


w_ = [10, 2, 5]  # original linear model weights to generate the data from and to compare later with w
noise = 5
x, t = get_data(w_, noise=noise, n=9)
y, w, RMSE = applyLST(x, t)
l = 1