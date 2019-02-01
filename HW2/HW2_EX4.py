import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import warnings
from HW2.HW2_EX2 import A
from HW2.HW2_EX2 import b
from HW2.HW2_EX2 import male_data
from HW2.HW2_EX2 import female_data
from HW2.HW2_EX3 import success_rate

A[:, 1] = A[:, 1] / 100


def theta_0_1():
    x = cp.Variable(A[0, :].size)
    objective = cp.Minimize(cp.sum_squares(A * x - np.reshape(b, b.size)) + 0.1 * cp.sum_squares(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob = cp.Problem(objective)
        prob.solve(solver='SCS')
    return x.value


def a_i_ii():
    lambda_list = np.arange(0.1, 10, 0.1)
    theta, rates = [], []
    first_part, last_part = [], []
    for lambda_ in lambda_list:
        x = cp.Variable(A[0, :].size)
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


def c_i():
    alpha_start = cp.sum_squares(theta_0_1()).value
    alpha_list = np.arange(alpha_start - 100, alpha_start + 102, 2)
    theta_alpha = []
    i = 0
    for alpha in alpha_list:
        i += 1
        print(str(i) + "/101")
        x = cp.Variable(A[0, :].size)
        expression = (cp.sum_squares(A * x - np.reshape(b, b.size)))
        constraint = [cp.sum_squares(x) <= alpha]
        objective = cp.Minimize(expression)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = cp.Problem(objective, constraint)
            prob.solve(solver=cp.SCS)
        theta_alpha.append(x.value)

    first_part_alpha, last_part_alpha = [], []
    alpha_x = []
    for i in range(len(alpha_list)):
        first_part_alpha.append(cp.sum_squares(np.matmul(A, theta_alpha[i]) - np.reshape(b, b.size)).value)
        last_part_alpha.append(cp.sum_squares(theta_alpha[i]).value)
        alpha_x.append(alpha_list[i])

    plt.figure(figsize=(6, 8))
    plt.subplot(311)
    plt.title(r'$||\theta_\alpha||_2^2\ and \ ||A\theta_\alpha - b||_2^2$')
    plt.xlabel(r'$||\theta_\alpha||_2^2$')
    plt.ylabel(r'$||A\theta_\alpha - b||_2^2$')
    plt.plot(last_part_alpha, first_part_alpha, c='r')
    plt.subplot(312)
    plt.title(r'$\alpha\ and \ ||\theta_\alpha||_2^2$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$||\theta_\alpha||_2^2$')
    plt.plot(alpha_x, last_part_alpha, c='g')
    plt.subplot(313)
    plt.title(r'$\alpha\ and \ ||A\theta_\alpha - b||_2^2$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$||A\theta_\alpha - b||_2^2$')
    plt.plot(alpha_x, first_part_alpha, c='b')
    plt.tight_layout()
    plt.savefig("screenshot/5.png", transparent=True, dpi=500, pad_inches=0)
    plt.show()


def c_ii():
    epsilon_start = cp.sum_squares(np.matmul(A, theta_0_1()) - np.reshape(b, b.size)).value
    epsilon_list = np.arange(epsilon_start, epsilon_start + 102, 2)
    theta_epsilon = []
    i = 0
    for epsilon in epsilon_list:
        i += 1
        print(str(i) + "/51")
        x = cp.Variable(A[0, :].size)
        expression = (cp.sum_squares(x))
        constraint = [cp.sum_squares(A * x - np.reshape(b, b.size)) <= epsilon]
        objective = cp.Minimize(expression)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = cp.Problem(objective, constraint)
            prob.solve(solver=cp.SCS)
        theta_epsilon.append(x.value)

    first_part_epsilon, last_part_epsilon = [], []
    epsilon_x = []
    for i in range(len(epsilon_list)):
        first_part_epsilon.append(cp.sum_squares(theta_epsilon[i]).value)
        last_part_epsilon.append(cp.sum_squares(np.matmul(A, theta_epsilon[i]) - np.reshape(b, b.size)).value)
        epsilon_x.append(epsilon_list[i])

    plt.figure(figsize=(6, 8))
    plt.subplot(311)
    plt.title(r'$||A\theta_\epsilon - b||_2^2\ and \ ||\theta_\epsilon||_2^2$')
    plt.xlabel(r'$||A\theta_\epsilon - b||_2^2$')
    plt.ylabel(r'$||\theta_\epsilon||_2^2$')
    plt.plot(last_part_epsilon, first_part_epsilon, c='r')
    plt.subplot(312)
    plt.title(r'$\epsilon\ and \ ||\theta_\epsilon||_2^2$')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$||\theta_\epsilon||_2^2$')
    plt.plot(epsilon_x, last_part_epsilon, c='g')
    plt.subplot(313)
    plt.title(r'$\epsilon\ and \ ||A\theta_\epsilon - b||_2^2$')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$||A\theta_\epsilon - b||_2^2$')
    plt.plot(epsilon_x, first_part_epsilon, c='b')
    plt.tight_layout()
    plt.savefig("screenshot/6.png", transparent=True, dpi=500, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # a_i_ii()
    c_i()
    # c_ii()
