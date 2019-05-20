import warnings

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy import optimize


def load_csv_data(filename):  # load data from given csv file to a numpy array
    a = []
    reader = np.genfromtxt(filename, delimiter=',')
    for row in reader:
        a.append(row)
    return np.array(a)


def logistic(X, labels, lr=0.05, max_epoch=100, mode='online'):
    """
    logistic algorithm for finding weight vector of classifier.
    In this implementation, it is assumed that samples in X is augmented by 1's if bias/offset
    is desired (i.e. bias/offset will not be calculated and returned separately by this function)
    :param X: (numpy.ndarray) 2D data array. Should have shape (K,N); K features, N samples
    :param labels: (numpy.ndarray) 1D label array. Shape (N,)
    :param lr: (number) learning rate for SGD/GD algorithm
    :param max_epoch: (int) maximum number of epochs for SGD/GD algorithm
        For each epoch during optimization, every sample in the data set is visited exactly once
    :param mode: (string) support ['online','batch']. Mode of optimization
        In 'online' mode, decision variable is updated after visiting one single sample
        In 'batch' mode, decision variable is updated after all samples in data set is visited
    :return:weight: (numpy.ndarray) 2D array of shape (K,1). Weight vector of classifier
    """
    if X.shape[1] != labels.size:
        print(X.shape[1])
        print(labels.size)
        raise ValueError('Number of samples is different from number of labels')
    if mode not in {'online', 'batch'}:
        raise ValueError('unknown mode ' + mode)

    def sigmoid(ary):
        return 1 / (1 + np.exp(-ary))

    w = np.random.randn(1, X.shape[0])
    if mode == 'online':
        for epoch in range(max_epoch):
            shuffled_index = np.random.permutation(X.shape[1])
            X = X[:, shuffled_index]
            labels = labels[shuffled_index]
            for i, label in enumerate(labels):
                w -= lr * ((sigmoid(w * X[:, i:i + 1]) - label) * X[:, i:i + 1]).T
            print(epoch)
    elif mode == 'batch':
        for epoch in range(max_epoch):
            w -= lr * ((sigmoid(w * X) - labels) * X).sum(axis=1, keepdims=True).T / X.shape[1]
    return w.T


def perceptron(X, labels, lr=0.05, max_epoch=100, mode='online'):
    """
    perceptron algorithm for finding weight vector and bias of classifier.
    It is assumed in this implementation that samples in X are not appended by 1's for accommodating offset/bias

    :param X: (numpy.ndarray) 2D data array. Should have shape (K,N); K features, N samples
    :param labels: (numpy.ndarray) 1D label array. Shape (N,)
    :param lr: (number) learning rate for SGD/GD algorithm
    :param max_epoch: (int) maximum number of epochs for SGD/GD algorithm
        For each epoch during optimization, every sample in the data set is visited exactly once
    :param mode: (string) support ['online','batch']. Mode of optimization
        In 'online' mode, decision variable is updated after visiting one single sample
        In 'batch' mode, decision variable is updated after all samples in data set is visited
    :return: weight: (numpy.ndarray) 2D array of shape (K,1). Weight vector of classifier
    :return: bias: (number) bias/offset of classifier
    """
    if X.shape[1] != labels.size:
        raise ValueError('Number of samples is different from number of labels')
    weight = np.zeros((1, X.shape[0]))
    bias = 0
    if mode == 'online':
        for epoch in range(max_epoch):
            num_false_in_epoch = 0
            shuffled_index = np.random.permutation(labels.size)
            X = X[:, shuffled_index]
            labels = labels[shuffled_index]
            for i, label in enumerate(labels):
                if label * (weight @ X[:, i:i + 1] + bias) <= 0:
                    num_false_in_epoch += 1
                    weight += lr * label * X[:, i]
                    bias += lr * label
            if not num_false_in_epoch:
                break
    elif mode == 'batch':
        for epoch in range(max_epoch):
            y = weight @ X + bias
            loss_each_sample = (-labels * y).squeeze()
            if any(loss_each_sample >= 0):
                weight += lr * (labels[loss_each_sample >= 0] * X[:, loss_each_sample >= 0]).sum(axis=1)
                bias += lr * labels[loss_each_sample >= 0].sum()
            else:
                break
    return weight.T, bias


def svm_cvxpy(X, labels, allow_slackness=False, c=1):
    """
    Implementation of Support Vector Machine with cvxpy optimization library
    It is assumed that offset/bias term for classifier is dealt with explicitly outside of
    this function by appending original data vectors with 1's, if offset/bias is desired
    (i.e. bias/offset will not be calculated and returned separately by this function)
    :param X: (numpy.ndarray) 2D data array. Should have shape (K,N); K features, N samples
    :param labels: (numpy.ndarray) 1D label array. Shape (N,)
    :param allow_slackness: (bool) whether slackness is allowed for the decision boundary
    :param c: (number) coefficient that controls the influence of slackness variables on objective function
    :return: weight: (numpy.ndarray) 2D array representing weight vector. Shape (K,1)
    :return: [slackness]: (numpy.ndarray) 1D array storing slackness for each sample.
        Returned only when allow_slackness set to True
    """
    if labels.ndim != 1:  # Number of array dimensions
        raise ValueError('Argument labels should be a 1D numpy ndarray')
    if X.shape[1] != labels.size:
        raise ValueError('Number of samples is different from number of labels')
    theta = cp.Variable((X.shape[0], 1))  # weight vector
    if allow_slackness:
        eps = cp.Variable((1, X.shape[1]))
        objective = cp.Minimize(cp.norm(theta) + c * cp.norm(eps, 1))
        constraints = [cp.multiply(np.expand_dims(labels, 0), theta.T @ X) >= 1 - eps, eps >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        print(theta.value)
        print(eps.value)
        return theta.value, eps.value
    else:
        objective = cp.Minimize(cp.norm(theta))
        constraints = [cp.multiply(np.expand_dims(labels, 0), theta.T @ X) >= 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = cp.Problem(objective, constraints)
            prob.solve()
        print(theta.shape)
        print(theta.value)
        return theta.value


def svm_scipy(X, labels, allow_slackness=False, c=1):
    """
    Implementation of SVM with scipy.optimize optimization libarary.
    It is assumed that offset/bias term for classification is dealt with explicitly outside
    of this function by appending original data vectors with 1's, if offset/bias is desired
    (i.e. bias/offset will not be calculated and returned separately by this function).
    :param X: (numpy.ndarray) 2D data array. Should have shape (K,N); K features, N samples
    :param labels: (numpy.ndarray) 1D label array. Shape (N,)
    :param allow_slackness: (bool) whether slackness is allowed for the decision boundary
    :param c: (number) coefficient that controls the influence of slackness variables on objective function
    :return: weight: (numpy.ndarray) 2D array representing weight vector. Shape (K,1)
    :return: [slackness]: (numpy.ndarray) 1D array storing slackness for each sample.
        Returned only when allow_slackness set to True
    """
    if X.shape[1] != labels.size:
        raise ValueError('Number of samples is different from number of labels')
    theta = np.random.randn(X.shape[0])  # weight vector
    if allow_slackness:
        # slackness for each data point
        eps = np.zeros(X.shape[1])
        # decision variables of primal SVM objective function
        decision_vars = np.hstack((theta, eps))

        # objective function that returns objective value 0.5*||theta||^2+c*sum(eps)
        def obj_func(dec_vars, *args):
            n_features = args[0]
            c = args[1]
            return 0.5 * np.sum(dec_vars[:n_features] * dec_vars[:n_features]) + c * np.sum(dec_vars[n_features])

        # returns Jacobian/gradient of objective function
        def obj_func_jac(dec_vars, *args):
            n_features = args[0]
            c = args[1]
            return np.hstack((dec_vars[:n_features], c * np.ones(dec_vars.size - n_features)))

        # returns value of each constraint evaluated at current point
        def constraint_func(dec_vars, *args):
            theta = dec_vars[:args[0]]
            eps = dec_vars[args[0]:]
            data_mat = args[1]
            labels = args[2]
            return np.hstack((np.squeeze(theta @ data_mat * labels - (1 - eps)), eps))

        constraints = {'type': 'ineq', 'fun': constraint_func, 'args': (theta.size, X, labels)}
        res = optimize.minimize(obj_func, decision_vars, args=(theta.size, c), method='SLSQP', jac=obj_func_jac,
                                constraints=constraints)
        if res.success:
            return np.expand_dims(res.x[:theta.size], 1), np.expand_dims(res.x[theta.size:], 0)
        else:
            raise RuntimeError('Optimizer fails to exits successfully. Solution was not found.')
    else:
        decision_vars = theta  # decision variables of primal SVM objective function

        # objective function that returns objective value 0.5*||theta||^2
        def obj_func(dec_vars, *args):
            return 0.5 * np.sum(dec_vars * dec_vars)

        # returns Jacobian/gradient of objective function
        def obj_func_jac(dec_vars, *args):
            return dec_vars

        # returns value of each constraint evaluated at current point
        def constraint_func(dec_vars, *args):
            theta = dec_vars
            data_mat = args[0]
            labels = args[1]
            return np.squeeze(theta @ data_mat * labels - 1)

        constraints = {'type': 'ineq', 'fun': constraint_func, 'args': (X, labels)}
        res = optimize.minimize(obj_func, decision_vars, method='SLSQP', jac=obj_func_jac, constraints=constraints)
        if res.success:
            return np.expand_dims(res.x, 1)
        else:
            raise RuntimeError('Optimizer fails to exits successfully. Solution was not found.')


if __name__ == '__main__':
    sample = load_csv_data("data/hw04_sample_vectors.csv").T
label = load_csv_data("data/hw04_labels.csv")
print(sample)
print(label)
# w = logistic(sample, label)
# q = logistic(sample, label, mode='batch')
# w = perceptron(sample, label, max_epoch=500)
q = perceptron(sample, label, max_epoch=5000, mode='batch')
w = svm_cvxpy(sample, label)
# q = svm_cvxpy(sample, label, allow_slackness=True)
# print(w)
print(q[0])
plt.scatter(sample[0, :1000], sample[1, :1000], marker='.', c='c', alpha=0.5)
plt.scatter(sample[0, 1000:], sample[1, 1000:], marker='.', c='r', alpha=0.5)
x0 = np.linspace(-0.5, 0.5, 100)
# x1 = w[0] * x0 + w[1]  # logistic
# x2 = q[0] * x0 + q[1]  # logistic
# x1 = -w[1] - w[0][0] * x0 / w[0][1]  # perceptron
x2 = -q[1] / q[0][0] - q[0][1] * x0  # perceptron
# plt.plot(x0, x1)
# plt.figure()
plt.plot(x0, x2)
plt.show()
