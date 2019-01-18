import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# x, y = np.mgrid[-1:5:.01, 0:10:.01]
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x
# pos[:, :, 1] = y
# z = multivariate_normal([2., 6.], [[2., 1.], [1., 2.]])
# plt.contour(x, y, z.pdf(pos))
# plt.savefig("6.png", transparent=True, dpi=500, pad_inches=0)
# plt.show()

z = np.random.multivariate_normal([2, 6], [[2, 1], [1, 1]], 5000)
plt.scatter(z[:, 0], z[:, 1], marker='.')
plt.show()
