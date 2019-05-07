import matplotlib.pyplot as plt
import numpy as np
from HW2.HW2_EX1 import load_csv_data
from HW2.HW2_EX2 import male_data
from HW2.HW2_EX2 import female_data
from HW2.HW2_EX2 import theta_np


# divide: True divide matrix A by 100, False not
def success_rate(theta, divide=False):
    male_test = load_csv_data("data/male_test_data.csv")[:, 1:3]
    female_test = load_csv_data("data/female_test_data.csv")[:, 1:3]
    if divide:
        male_test[:, 1] = male_test[:, 1] / 100
        female_test[:, 1] = female_test[:, 1] / 100
    total = 0
    s_m, s_f = 0, 0
    for male in male_test:
        g_x = male[0] * theta[0] + male[1] * theta[1] + theta[2]
        if np.sign(g_x) == 1:
            s_m += 1
        total += 1
    for female in female_test:
        g_x = female[0] * theta[0] + female[1] * theta[1] + theta[2]
        if np.sign(g_x) == -1:
            s_f += 1
        total += 1
    return (s_m + s_f) / total


if __name__ == '__main__':
    print("Success rate:")
    print(success_rate(theta_np))
    plt.scatter(male_data[:, 0], male_data[:, 1], marker='.', c='', edgecolors='r', alpha=0.4)
    plt.scatter(female_data[:, 0], female_data[:, 1], marker='.', c='', edgecolors='c', alpha=0.4)
    x1 = np.linspace(10, 80, 100)
    x2 = (-theta_np[2] - theta_np[0] * x1) / theta_np[1]
    plt.plot(x1, x2)
    plt.savefig("screenshot/1.png", transparent=True, pad_inches=0)
    plt.show()
