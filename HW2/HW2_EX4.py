import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import warnings
from HW2.HW2_EX2 import A
from HW2.HW2_EX2 import b
from HW2.HW2_EX2 import male_data
from HW2.HW2_EX2 import female_data
from HW2.HW2_EX3 import success_rate

lambda_list = np.arange(0.1, 10, 0.1)
theta, rates = [], []
first_part, last_part = [], []
A[:, 1] = A[:, 1] / 100
for lambda_ in lambda_list:
    x = cp.Variable(A[0, :].size)

    # This is not correct, all the plots are almost the same
    # objective = cp.Minimize(cp.square(cp.norm(A * x - np.reshape(b, b.size))) + l * cp.square(cp.norm(x)))

    # This is the correct expression of the formula
    objective = cp.Minimize(cp.sum_squares(A * x - np.reshape(b, b.size)) + lambda_ * cp.sum_squares(x))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob = cp.Problem(objective)
        prob.solve(solver='SCS')
    theta.append(x.value)
    rate = success_rate(x.value, divide=True)
    rates.append(rate)
    print("Lambda: %.2f" % lambda_ + "\tsuccess rate: " + str(rate))

x = np.linspace(10, 80, 100)
legend_str = []
lambda_x = []
plt.figure(1, figsize=(15, 7.5))
for i in range(len(lambda_list))[0::10]:
    y = (-theta[i][2] - theta[i][0] * x) / theta[i][1]
    first_part.append(cp.sum_squares(np.matmul(A, theta[i]) - np.reshape(b, b.size)).value)
    last_part.append(cp.sum_squares(theta[i]).value)
    lambda_x.append(lambda_list[i])
    legend_str.append('$\lambda=$' + str(lambda_list[i]))
    plt.plot(x, y)

plt.legend(legend_str)
plt.scatter(male_data[:, 0], male_data[:, 1] / 100, marker='.', c='', edgecolors='r', alpha=0.4)
plt.scatter(female_data[:, 0], female_data[:, 1] / 100, marker='.', c='', edgecolors='c', alpha=0.4)
# plt.scatter(male_data[:, 0], male_data[:, 1], marker='.', c='', edgecolors='r', alpha=0.4)
# plt.scatter(female_data[:, 0], female_data[:, 1], marker='.', c='', edgecolors='c', alpha=0.4)
plt.xlabel('BMI')
plt.ylabel('Stature')
plt.tight_layout()
plt.savefig("screenshot/2.png", transparent=True, dpi=500, pad_inches=0)
plt.figure(2)
plt.xlabel(r'$\lambda$')
plt.ylabel('success rate')
plt.plot(lambda_list, rates)
plt.tight_layout()
plt.savefig("screenshot/3.png", transparent=True, dpi=500, pad_inches=0)
plt.figure(3, figsize=(6, 8))
plt.subplot(311)
plt.title(r'$||\theta_\lambda||_2^2\ and \ ||A\theta_\lambda - b||_2^2$')
plt.xlabel(r'$||\theta_\lambda||_2^2$')
plt.ylabel(r'$||A\theta_\lambda - b||_2^2$')
plt.plot(last_part, first_part, c='r')
plt.subplot(312)
plt.title(r'$\lambda\ and \ ||\theta_\lambda||_2^2$')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$||\theta_\lambda||_2^2$')
plt.plot(lambda_x, last_part, c='g')
plt.subplot(313)
plt.title(r'$\lambda\ and \ ||A\theta_\lambda - b||_2^2$')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$||A\theta_\lambda - b||_2^2$')
plt.plot(lambda_x, first_part, c='b')
plt.tight_layout()
plt.savefig("screenshot/4.png", transparent=True, dpi=500, pad_inches=0)
plt.show()
