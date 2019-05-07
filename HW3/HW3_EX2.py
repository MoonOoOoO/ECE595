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


# return 1 if it's cat, 0 if it's grass
def decision_func(x):
    if discriminant_cat(x) > discriminant_grass(x):
        label = 1
    else:
        label = 0
    return label


def cal_loss(output):
    loss = 0
    truth = plt.imread('data/truth.png') / 255
    x, y = truth.shape
    count = 0
    for k in range(x - 8):
        for l in range(y - 8):
            loss += np.abs(output[k, l] - truth[k, l])
            count += 1
    return loss / count


# main function
if __name__ == '__main__':
    # img = plt.imread('data/cat_grass.jpg') / 255
    # m, n = img.shape
    img = plt.imread('data/cat_on_grass.jpg') / 255
    m, n, c = img.shape
    img = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / c
    output_overlap = np.zeros((m - 8, n - 8))
    output_non_overlap = np.zeros((m - 8, n - 8))
    process = 0
    for i in range(m - 8):
        for j in range(n - 8):
            # non-overlap
            if (i % 8 == 0) and (j % 8 == 0):
                patch_non_overlap = img[i:i + 8, j:j + 8]
                patch_non_overlap = np.reshape(patch_non_overlap, (64, 1))
                output_non_overlap[i:i + 8, j:j + 8] = decision_func(patch_non_overlap)
            # overlap
            patch_overlap = img[i:i + 8, j:j + 8]
            patch_overlap = np.reshape(patch_overlap, (64, 1))
            output_overlap[i, j] = decision_func(patch_overlap)
            print(process / ((m - 8) * (n - 8)))
            process += 1

    # display overlap mode
    plt.figure()
    plt.axis('off')
    # loss_overlap = cal_loss(output_overlap)
    # plt.title('Loss: ' + str(loss_overlap))
    plt.title('cheetah on grass, overlap')
    plt.imshow(output_overlap * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/5.png", transparent=True, pad_inches=0)

    # display non-overlap mode
    plt.figure()
    plt.axis('off')
    # loss_non_overlap = cal_loss(output_non_overlap)
    # plt.title('Loss: ' + str(loss_non_overlap))
    plt.title('cheetah on grass, non-overlap')
    plt.imshow(output_non_overlap * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/6.png", transparent=True, pad_inches=0)
    plt.show()
