from HW3.HW3_EX2 import cat_cov, grass_cov, cat_mean, grass_mean
import numpy as np
import matplotlib.pyplot as plt


# target_label 1 for cat, 0 for grass, this one misclassifies cat into grass
def gradient(patch_0, patch_x, _lambda, target_label, label):
    grad = np.reshape(np.zeros(64), (64, 1))
    # label = decision_func(patch_0)
    if target_label != label:  # if input patch is not target label
        cat_cov_inv = np.linalg.inv(cat_cov)
        grass_cov_inv = np.linalg.inv(grass_cov)
        grad = 2 * (patch_x - patch_0) + _lambda * (np.matmul(cat_cov_inv - grass_cov_inv, patch_x) +
                                                    np.matmul(cat_cov_inv, cat_mean) -
                                                    np.matmul(grass_cov_inv, grass_mean))
    return grad


# target_label 1 for cat, 0 for grass
def cw_attack(image, _lambda, target_label, alpha):
    m, n = image.shape
    pert_img_curr = np.copy(image)
    itr_num, change = 1, 1
    changes = []
    label = plt.imread('data/truth.png')
    while itr_num <= 300 and change >= 0.01:
        pert_img_prev = np.copy(pert_img_curr)
        for i in range(m - 8):
            for j in range(n - 8):
                # non-overlap
                patch_0 = image[i:i + 8, j:j + 8]  # x_0
                patch_0 = np.reshape(patch_0, (64, 1))
                patch = pert_img_prev[i:i + 8, j:j + 8]  # x_k-1
                patch = np.reshape(patch, (64, 1))
                # launch the attack
                grad = gradient(patch_0, patch, _lambda, target_label, label[i, j])
                pert_img_curr[i:i + 8, j:j + 8] = np.reshape(np.clip((patch - alpha * grad), 0.0, 1.0), (8, 8))  # X_k
        change = np.linalg.norm(pert_img_curr - pert_img_prev)
        changes.append(change)
        print("Iteration " + str(itr_num) + ", change = " + str(change))
        itr_num += 1
    return pert_img_curr, changes


if __name__ == '__main__':
    img = plt.imread('data/cat_grass.jpg') / 255
    print("lambda = 1")
    cw_img_1, changes_1 = cw_attack(img, 1, 1, 0.0001)
    print("lambda = 0.5")
    cw_img_0_5, changes_0_5 = cw_attack(img, 0.5, 1, 0.0001)
    print("lambda = 5")
    cw_img_5, changes_5 = cw_attack(img, 5, 1, 0.0001)

    plt.figure()
    plt.imshow(cw_img_0_5 * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/0_5_overlap.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    plt.imshow((cw_img_0_5 - img) * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/0_5_overlap_perturbation.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    plt.imshow(cw_img_1 * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/1_overlap.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    plt.imshow((cw_img_1 - img) * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/1_overlap_perturbation.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    plt.imshow(cw_img_5 * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/5_overlap.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    plt.imshow((cw_img_5 - img) * 255, cmap='gray')
    plt.tight_layout()
    plt.savefig("screenshot/5_overlap_perturbation.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    c1_x = np.arange(1, len(changes_0_5) + 1, 5)
    c1_y = changes_0_5[1:len(changes_0_5) + 1:5]
    plt.plot(c1_x, c1_y, 'o', c1_x, c1_y)
    plt.tight_layout()
    plt.savefig("screenshot/1_overlap_plot.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    c5_x = np.arange(1, len(changes_1) + 1, 5)
    c5_y = changes_1[1:len(changes_1) + 1:5]
    plt.plot(c5_x, c5_y, 'o', c5_x, c5_y)
    plt.tight_layout()
    plt.savefig("screenshot/5_overlap_plot.png", transparent=True, dpi=500, pad_inches=0)

    plt.figure()
    c10_x = np.arange(1, len(changes_5) + 1, 2)
    c10_y = changes_5[1:len(changes_5) + 1:2]
    plt.plot(c10_x, c10_y, 'o', c10_x, c10_y)
    plt.tight_layout()
    plt.savefig("screenshot/10_overlap_plot.png", transparent=True, dpi=500, pad_inches=0)
    plt.show()
