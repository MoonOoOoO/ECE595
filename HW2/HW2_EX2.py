import numpy as np
import cvxpy as cp
import warnings
from HW2.HW2_EX1 import load_csv_data

# load male training data
male_data = load_csv_data("data/male_train_data.csv")[:, 1:3]
male_label = np.ones((male_data[:, 0].size, 1))
male_set = np.concatenate((male_data, male_label), axis=1)

# load female training data
female_data = load_csv_data("data/female_train_data.csv")[:, 1:3]
female_label = -1 * np.ones((female_data[:, 0].size, 1))
female_set = np.concatenate((female_data, female_label), axis=1)

female_set_temp = np.array(female_set)
female_set_temp[:, 2] = -1 * female_set_temp[:, 2]

# construct matrix A and b
A = np.concatenate((male_set, female_set_temp), axis=0)
b = np.concatenate((male_label, female_label), axis=0)

# compute theta using numpy
temp = np.linalg.inv(np.matmul(A.T, A))
theta_np = np.matmul(np.matmul(temp, A.T), b)

# compute theta using cvxpy
theta_cp = cp.Variable(A[0, :].size)
objective = cp.Minimize(cp.sum_squares(A * theta_cp - np.reshape(b, b.size)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prob = cp.Problem(objective)
    prob.solve()

if __name__ == '__main__':
    print("Result from Numpy:")
    print(np.reshape(theta_np, theta_np.size))
    print("Result from CVXPY:")
    print(theta_cp.value)
