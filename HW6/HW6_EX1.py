import numpy as np
import matplotlib.pyplot as plt

num_exp = 100000  # number of experiments
num_coin = 1000  # number of coins
N = 10  # number of flips
mu = 0.5
epsilon_list = np.arange(0, 0.5, 0.05)


def calculate_P(v_list, mu, eps):
    total = 0
    for v in v_list:
        if np.abs(v - mu) > eps:
            total += 1
    return total / num_exp


first_coin = []
rand_coin = []
min_coin = []
for i in range(num_exp):
    flips = np.random.choice((0, 1), size=(num_coin, N))

plt.legend(['$v_1', '$v_{rand}', '$v_{min}'])
plt.show()
