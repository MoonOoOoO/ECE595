import math
import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(-50, 50, 0.1)
x2 = np.arange(10, 100, 0.1)
X1, X2 = np.meshgrid(x1, x2)
z = np.exp((-1 * X1 ** 2 - 2 * X1 + X1 * X2 + X2 ** 2 - 28 + 10 * X2) / 3) / (2 * np.sqrt(3))

plt.contour(x1, x2, z)
plt.show()
