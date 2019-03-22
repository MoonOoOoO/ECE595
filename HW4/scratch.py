import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt


def load_csv_data(filename):  # load data from given csv file to a numpy array
    a = []
    reader = np.genfromtxt(filename, delimiter=',')
    for row in reader:
        a.append(row)
    return np.array(a)


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def logistic_loss(y, y_hat):
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def logistic(X, labels, learning_rate, max_num_iterations):
    w = np.zeros((2, 1))
    b = 0
    for k in max_num_iterations:
        Z = np.matmul(X, w) + b
        A = sigmoid(Z)
        loss = logistic_loss(A, labels)
        dz = A - labels
        dw = 1 / 2000 * np.matmul(X, dz)
        db = np.sum(dz)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b


sample = load_csv_data("data/hw04_sample_vectors.csv")
label = load_csv_data("data/hw04_labels.csv")
logistic_model = LogisticRegression(solver="sag")
logistic_model.fit(sample, label)
per_model = Perceptron()
per_model.fit(sample, label)
hard_svm_model = LinearSVR(C=100)
hard_svm_model.fit(sample, label)
soft_svm_model = LinearSVR()
soft_svm_model.fit(sample, label)
print(logistic_model.coef_)
print(logistic_model.intercept_)
print(logistic_model.n_iter_)
print hard_svm_model.coef_

plt.scatter(sample[:1000, 0], sample[:1000, 1], marker='.', c='', edgecolors='c', alpha=0.5)
plt.scatter(sample[1000:, 0], sample[1000:, 1], marker='.', c='', edgecolors='r', alpha=0.5)
x0 = np.linspace(-0.5, 0.5, 100)
x1 = (-logistic_model.intercept_ - logistic_model.coef_[0][0] * x0) / logistic_model.coef_[0][1]
x2 = (-per_model.intercept_ - per_model.coef_[0][0] * x0) / per_model.coef_[0][1]
x3 = (-hard_svm_model.intercept_ - hard_svm_model.coef_[0] * x0) / hard_svm_model.coef_[1]
x4 = (-soft_svm_model.intercept_ - soft_svm_model.coef_[0] * x0) / soft_svm_model.coef_[1]
plt.plot(x0, x1, c='black')
plt.plot(x0, x2, c='b')
plt.plot(x0, x3, c='g')
plt.plot(x0, x4, c='c')
plt.show()
