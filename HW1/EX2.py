import math

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv
import imageio

from scipy.stats import norm

# mu = 0
# sigma = 1
# x = np.linspace(-3, 3, 100)
# y = np.exp(-(x - mu) ** 2 / 2 * sigma ** 2) / sigma * np.sqrt(2 * math.pi)
# plt.plot(x, y, "r-", linewidth=2)
# plt.savefig("3.png", transparent=True, dpi=500, pad_inches=0)
# plt.show()

# x = np.random.normal(0, 1, 1000)
# mu, sigma = norm.fit(x)
# print(mu, sigma)
# plt.figure(1)
# plt.subplot(211)
# a = plt.hist(x, 4, density=True)
# plt.plot(a[1], norm.pdf(a[1]), "r")
# plt.subplot(212)
# b = plt.hist(x, 1000, density=True)
# plt.plot(b[1], norm.pdf(b[1]), "r")
# plt.savefig("4.png", transparent=True, dpi=500, pad_inches=0)
# plt.show()

x = np.random.normal(0, 1, 1000)
x.sort()
d = x.max() - x.min()
j_h = 0
label = 0
j_ha = np.zeros(200)
for m in range(1, 201):
    hist, bins = np.histogram(x, m)
    h = d / m
    s = 0
    for item in hist:
        s += (item / 1000) ** 2
    temp = 2 / (h * 999) - s * 1001 / (h * 999)
    if m == 0:
        j_h = temp
    elif temp < j_h:
        label = m
        j_h = temp
    j_ha[m - 1] = j_h

print(label)
step = np.linspace(1, 200, 200)
plt.plot(step, j_ha)
# p = plt.hist(x, label, density=True)
# plt.plot(p[1], norm.pdf(p[1]), "r")
plt.savefig("jha.png", transparent=True, dpi=500, pad_inches=0)
plt.show()
