import matplotlib.pyplot as plt
import numpy as np


def minimiser(a):
    x = np.linspace(a - 10, a + 10, 31)
    w0 = 1
    w1 = 8
    y = w0 + w1 * (x - a) ** 2
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xticks(np.arange(x.min(), x.max(), step=5))
    plt.yticks(np.arange(y.min() - 1, y.max(), step=200))
    plt.plot(x, y, '-r')
    plt.grid()
    plt.show()
    print(x[np.argmin(y)] == a)


for a in range(-10, 10):
    minimiser(a)

