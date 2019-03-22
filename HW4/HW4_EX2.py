import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def load_csv_data(filename):  # load data from given csv file to a numpy array
    a = []
    reader = np.genfromtxt(filename, delimiter=',')
    for row in reader:
        a.append(row)
    return np.array(a)


#
# def logistic(X, labels, learning_rate, max_num_iterations):
#     for m in range(max_num_iterations):
#         theta -= learning_rate * cost_function_derivative(X, labels, ???)
#     return theta

sample = load_csv_data("data/hw04_sample_vectors.csv")
label = load_csv_data("data/hw04_labels.csv")
model = LogisticRegression(solver="lbfgs")
model.fit(sample, label)
print(model.coef_)
print(model.intercept_)

plt.scatter(sample[:1000, 0], sample[:1000, 1], marker='.', c='', edgecolors='c', alpha=0.5)
plt.scatter(sample[1000:, 0], sample[1000:, 1], marker='.', c='', edgecolors='r', alpha=0.5)
x1 = np.linspace(-0.5, 0.5, 100)
x2 = (-model.intercept_ - model.coef_[0][0] * x1) / model.coef_[0][1]
plt.plot(x1, x2, c='black')
plt.show()
