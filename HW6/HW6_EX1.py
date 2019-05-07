import numpy as np
import matplotlib.pyplot as plt

N = 10  # flip each coin 10 times
num_coin = 1000  # 1000 coins
num_exp = 100000  # number of experiments, equals to 1000x10
epsilon_list = np.arange(0, 0.5, 0.05)  # epsilon value


def calculate_p(v_list, epsilon):
    total = 0
    mu = 0.5
    for v in v_list:
        if np.abs(v - mu) > epsilon:
            total += 1
    return total / num_exp


first_coin = []
rand_coin = []
min_coin = []
for i in range(num_exp):
    flips = np.random.choice((0, 1), size=(num_coin, N))
    head_frequency = np.mean(flips, axis=1)
    first_coin.append(head_frequency[0])
    rand_coin.append(head_frequency[np.random.randint(1000)])
    min_coin.append(np.min(head_frequency))
    print((i + 1) / num_exp)

plt.figure(figsize=(5, 8))
plt.subplot(311)
plt.title("first coin")
plt.hist(first_coin, range=(0, 1), color='r')
plt.subplot(312)
plt.title("random coin")
plt.hist(rand_coin, range=(0, 1), color='gray')
plt.subplot(313)
plt.title("minimum coin")
plt.hist(min_coin, range=(0, 1), color='c')
plt.tight_layout()
plt.savefig("screenshot/1.png", transparent=True, pad_inches=0)

P_first_coin = []
P_rand_coin = []
P_min_coin = []
hoeffding = []
for eps in epsilon_list:
    P_first_coin.append(calculate_p(first_coin, eps))
    P_rand_coin.append(calculate_p(rand_coin, eps))
    P_min_coin.append(calculate_p(min_coin, eps))
    hoeffding.append(2 * np.exp(-2 * eps ** 2 * N))
plt.figure(figsize=(5, 8))
plt.subplot(411)
plt.title('$P(|v_1-\mu|>\epsilon)$')
plt.plot(epsilon_list, P_first_coin, 'o', epsilon_list, P_first_coin)
plt.xticks(epsilon_list)

plt.subplot(412)
plt.title('$P(|v_{rand}-\mu|>\epsilon)$')
plt.plot(epsilon_list, P_rand_coin, 'o', epsilon_list, P_rand_coin)
plt.xticks(epsilon_list)

plt.subplot(413)
plt.title('$P(|v_{min}-mu|>\epsilon)$')
plt.plot(epsilon_list, P_min_coin, 'o', epsilon_list, P_min_coin)
plt.xticks(epsilon_list)

plt.subplot(414)
plt.title('$2e^{-2\epsilon^2 N}$')
plt.plot(epsilon_list, hoeffding, 'o', epsilon_list, hoeffding)
plt.xticks(epsilon_list)

plt.tight_layout()
plt.savefig("screenshot/2.png", transparent=True, pad_inches=0)
plt.show()
