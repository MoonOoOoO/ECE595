import numpy as np
import matplotlib.pyplot as plt

# training data
train_cat = np.matrix(np.loadtxt("data/train_cat.txt", delimiter=',')).astype(np.float_)
train_grass = np.matrix(np.loadtxt("data/train_grass.txt", delimiter=',')).astype(np.float_)
# mean
cat_mean = np.mean(train_cat, axis=1)
grass_mean = np.mean(train_grass, axis=1)
# covariance
cat_cov = np.cov(train_cat)
grass_cov = np.cov(train_grass)
# prior
N = train_cat.shape[1] + train_grass.shape[1]
pi_cat = train_cat.shape[1] / N
pi_grass = 1 - pi_cat


def discriminant_cat(x):
    temp = np.matrix(x - cat_mean)
    pre = -np.matmul(np.matmul(temp.T, np.linalg.inv(cat_cov)), temp) / 2
    score = pre - np.math.log(np.linalg.det(cat_cov)) / 2 + np.math.log(pi_cat)
    return score.item(0)


def discriminant_grass(x):
    temp = np.matrix(x - grass_mean)
    pre = -np.matmul(np.matmul(temp.T, np.linalg.inv(grass_cov)), temp) / 2
    score = pre - np.math.log(np.linalg.det(grass_cov)) / 2 + np.math.log(pi_grass)
    return score.item(0)


def decision_func(x):
    if discriminant_cat(x) > discriminant_grass(x):
        label = 1
    else:
        label = 0
    return label


if __name__ == '__main__':
    img = plt.imread('data/cat_grass.jpg') / 255
    m, n = img.shape
    output = np.zeros((m - 8, n - 8))
    process = 0
    for i in range(m - 8):
        for j in range(n - 8):
            patch = img[i:i + 8, j:j + 8]
            patch = np.reshape(patch, (64, 1))
            output[i, j] = decision_func(patch)
            print(process / ((m - 8) * (n - 8)))
            process += 1
    plt.imshow(output * 255, cmap='gray')
    # plt.axis('off')
    # plt.imshow(Y, cmap='gray')
    plt.savefig("screenshot/1.png", transparent=True, dpi=500, pad_inches=0)
    plt.show()
