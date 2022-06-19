import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N = 70 #15

t = np.linspace(0, 2*np.pi, num=N)
a = 0.5
b = 0.5
r = 1
x = a + r * np.cos(t)
y = b + r * np.sin(t)

mn, mx = a-3*r, b+3*r
noise = 0.5*r

Y = np.random.normal(y, noise)
X = np.random.normal(x, noise)

k = 1