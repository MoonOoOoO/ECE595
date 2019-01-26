import csv
import numpy as np


# # Problem data.
# m = 30
# n = 20
# np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m)
#
# # Construct the problem.
# x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A * x - b))
# constraints = [0 <= x, x <= 1]
# prob = cp.Problem(objective, constraints)
#
# # The optimal objective value is returned by `prob.solve()`.
# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print(x.value)
# # The optimal Lagrange multiplier for a constraint is stored in
# # `constraint.dual_value`.
# print(constraints[0].dual_value)

def load_csv_data(filename):
    a = []
    index = True
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if index:
                index = False
            else:
                a.append(row)
    csv_file.close()
    return np.array(a)


male_train_data = load_csv_data("data/male_train_data.csv")
female_train_data = load_csv_data("data/female_train_data.csv")
print(male_train_data)
print(female_train_data)
