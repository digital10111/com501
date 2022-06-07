import matplotlib.pyplot as plt
import numpy as np

m = [40, 10]
a = m[0]
b = m[1]

fig = plt.figure()
ax = fig.gca(projection='3d')
step = 0.3

x = np.arange(a-10, a+10, step)
y = np.arange(b-10, b+10, step)

X, Y = np.meshgrid(x, y)

Z = (X-a)**2 + (Y-b)**2

surf = ax.plot_surface(X, Y, Z)

ax.set_xticks(np.arange(x.min(), x.max(), step=5))
ax.set_yticks(np.arange(y.min(), y.max(), step=5))
ax.set_zticks(np.arange(Z.min(), Z.max(), step=50))

ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')
ax.set_zlabel('$J(m_1)$')
plt.tight_layout()
plt.show()

eps = 0.01
ind = round(np.argmin(Z)/(len(x) + 1))
print(x[ind]-a < eps and y[ind]-b < eps)
k = 1