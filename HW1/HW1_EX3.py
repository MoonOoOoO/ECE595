import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import multivariate_normal

# x, y = np.mgrid[-1:5:.01, 0:10:.01]
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x
# pos[:, :, 1] = y
# z1 = multivariate_normal([2., 6.], [[2., 1.], [1., 2.]])
# plt.contour(x, y, z1.pdf(pos))
# z2 = np.random.multivariate_normal([2, 6], [[2, 1], [1, 1]], 5000)
# plt.scatter(z2[:, 0], z2[:, 1], marker='+', c='r')
# plt.savefig("4.png", transparent=True, dpi=500, pad_inches=0)
# plt.show()

x, y = np.random.multivariate_normal([0, 0], [[2, 1], [1, 2]], 5000).T
plt.scatter(x, y, marker='.', c='g', alpha=0.5)
eig_val, eig_vector = np.linalg.eig([[2, 1], [1, 2]])
lam = [[np.sqrt(3), 0], [0, 1]]
mat = np.dot(np.dot(eig_vector, lam), eig_vector.T)
inter = np.column_stack((x, y))
x_t, y_t = x, y
for i in range(0, 5000):
    trans = np.dot(mat, inter[i]) + [2, 6]
    x_t[i] = trans[0]
    y_t[i] = trans[1]

plt.scatter(x_t, y_t, marker='+', c='r', alpha=0.5)
plt.savefig("5.png", transparent=True, dpi=500, pad_inches=0)
plt.show()
