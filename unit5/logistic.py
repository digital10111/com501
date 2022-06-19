import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def sig(z):
    return 1/(1 + np.exp(-z))


x = np.linspace(-10, 10, num = 100)
plt.plot(x, sig(x))
plt.show()


def get_x2(x1, c=4):
    return 20 * x1 + c


def getData(n=10):
    x1 = np.linspace(0, 1, num=n)
    x2 = get_x2(x1)  # 2*x1 - 4 #12
    x2 = np.random.normal(x2, 2)  # get the second attribute (add noise so data is realistic)

    return x1, x2


def LinC12(x1, x2):
    LM = -x2 + get_x2(x1)  # 2*X1 - 4  # linear model decision boundary
    ind1 = LM >= 0
    ind0 = LM < 0

    C1 = [x1[ind1], x2[ind1]]  # Actual Class1
    C0 = [x1[ind0], x2[ind0]]  # Actual Class0

    targets = ind1 * 1.0  # targets can be deduced form the positive class

    return C1, C0, LM, targets


def plotC12(C0, C1, s=200):
    plt.scatter(C1[0], C1[1], s=s, color='b', marker='+', label='Class1')
    plt.scatter(C0[0], C0[1], s=s, color='none', marker='o', label='Class0', edgecolor='red')


X1, X2 = getData(100)
C1, C0, LM1, targets = LinC12(X1, X2) # separate the classes with a linear boundaries

plotC12(C0, C1)
plt.plot(X1, get_x2(X1), 'b')
plt.show()
k = 2