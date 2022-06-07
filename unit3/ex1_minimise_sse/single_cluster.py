import matplotlib.pyplot as plt
import numpy as np

n = 31
a = 5
x = np.linspace(a-10, a+10, n)
mu = x.mean()

m = np.linspace(x.min(), x.max(), n)
SSE = np.zeros(m.shape)

for i, mi in enumerate(m):
    SSE[i] = ((x-mi)**2).sum()

plt.plot(x, SSE, '-r')
plt.grid()
plt.show()

print(SSE.min())
print(SSE.argmin())
print(m[SSE.argmin()])
print(x.mean() == m[SSE.argmin()])