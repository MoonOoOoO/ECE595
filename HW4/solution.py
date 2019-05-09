import numpy as np
import matplotlib.pyplot as plt


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
                w -= lr * ((sigmoid(w @ X[:, i:i + 1]) - label) * X[:, i:i + 1]).T
            print(epoch)
    elif mode == 'batch':
        for epoch in range(max_epoch):
            w -= lr * ((sigmoid(w @ X) - labels) * X).sum(axis=1, keepdims=True).T / X.shape[1]
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
    :return: weight: (numpy.ndarrar) 2D array of shape (K,1). Weight vector of classifier
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


if __name__ == '__main__':
    sample = load_csv_data("data/hw04_sample_vectors.csv").T
    label = load_csv_data("data/hw04_labels.csv")
    # w = logistic(sample, label)
    # q = logistic(sample, label, mode='batch')
    w = perceptron(sample, label, max_epoch=500)
    q = perceptron(sample, label, max_epoch=500, mode='batch')
    print(w)
    print(q)
    plt.scatter(sample[0, :1000], sample[1, :1000], marker='.', c='c', alpha=0.5)
    plt.scatter(sample[0, 1000:], sample[1, 1000:], marker='.', c='r', alpha=0.5)
    x0 = np.linspace(-0.5, 0.5, 100)
    # x1 = w[0] * x0 + w[1]
    # x2 = q[0] * x0 + q[1]
    x1 = -w[1] - w[0][0] * x0 / w[0][1]
    x2 = -q[1] - q[0][0] * x0 / q[0][1]
    plt.plot(x0, x1)
    plt.figure()
    plt.plot(x0, x2)
    plt.show()
