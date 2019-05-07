import warnings
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# construct dataset
mu_1 = [0, 0]
mu_2 = [10, 10]
cov = [[3, 1], [1, 3]]
x_1, x_2 = np.random.multivariate_normal(mu_1, cov, 1000).T
y_1, y_2 = np.random.multivariate_normal(mu_2, cov, 1000).T
plt.text(3, -5, r'$\mu_1$=(0,0)', color='g')
plt.text(3, 15, r'$\mu_2$=(10,10)', color='c')
plt.text(-5, 8, r'cov=((3,1), (1,3))', color='black')
plt.scatter(x_1, x_2, marker='.', c='g', alpha=0.5)
plt.scatter(y_1, y_2, marker='.', c='c', alpha=0.5)

# use formula in (a)
beta = np.matmul(np.linalg.inv(cov), np.subtract(mu_2, mu_1))
beta_0 = - np.matmul(np.matmul(np.add(mu_1, mu_2).T, np.linalg.inv(cov)), np.subtract(mu_2, mu_1)) / 2
# plot the result
m = np.linspace(-5, 15, 100)
n = -beta[1] * m / beta[0] - beta_0 / beta[0]
plt.plot(m, n, color='r', alpha=0.5)

# use LLS CVX
x_set = np.array([x_1, x_2, np.ones(1000)]).T
y_set = np.array([y_1, y_2, np.ones(1000)]).T
A = np.concatenate((x_set, y_set), axis=0)
b = np.concatenate((np.ones((1000, 1)), -np.ones((1000, 1))))
theta_cp = cp.Variable(A[0, :].size)
objective = cp.Minimize(cp.sum_squares(A * theta_cp - np.reshape(b, b.size)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prob = cp.Problem(objective)
    prob.solve()
theta_cp = theta_cp.value
print(theta_cp)

# plot the result
m_LLS = np.linspace(-5, 15, 100)
n_LLS = (-theta_cp[2] - theta_cp[0] * m_LLS) / theta_cp[1]
plt.plot(m_LLS, n_LLS, color='b', alpha=0.5)
plt.legend(['LLS', 'Gaussian'])
plt.tight_layout()
plt.savefig("screenshot/7.png", transparent=True, pad_inches=0)
plt.show()
