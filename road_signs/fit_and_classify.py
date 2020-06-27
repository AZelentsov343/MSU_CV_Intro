import numpy as np
from scipy.signal import convolve2d
from math import sqrt
import sklearn.svm as svm
from skimage.transform import resize


def gradient(image, type_of_grad='diff'):
    gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float64)
    if type_of_grad == 'diff':
        x_kernel = [[1., 0., -1.]]
        y_kernel = [[1.],
                    [0.],
                    [-1.]]
    elif type_of_grad == 'sobel':
        x_kernel = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        y_kernel = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
    else:
        raise Exception(f'Unknown type of grad: {type_of_grad}')
    i_x = convolve2d(gray, x_kernel, boundary='symm', mode='same')
    i_y = convolve2d(gray, y_kernel, boundary='symm', mode='same')
    modules = np.sqrt(i_x ** 2 + i_y ** 2).astype(np.float64)
    directions = np.arctan2(i_y, i_x) + np.pi
    return directions, modules


def histogram(image, cell_rows, cell_cols, bin_count):
    grad_dir, grad_mod = gradient(image)

    angle_bin_size = 2 * np.pi / bin_count
    angle_bins = np.zeros((image.shape[0], image.shape[1], bin_count), dtype='float64')
    result = np.zeros((image.shape[0] // cell_rows, image.shape[1] // cell_cols, bin_count), dtype='float64')

    for i in range(bin_count):
        angle_bins[:, :, i] = np.logical_and(grad_dir >= i * angle_bin_size,
                                             grad_dir < i * angle_bin_size + angle_bin_size) * grad_mod

    for i in range(image.shape[0] // cell_rows):
        for j in range(image.shape[1] // cell_cols):
            for k in range(bin_count):
                result[i, j, k] = np.sum(angle_bins[i * cell_rows:i * cell_rows + cell_rows,
                                                    j * cell_cols:j * cell_cols + cell_cols, k])

    return result


def extract_hog(image):
    block_row_cells = 2
    block_col_cells = 2
    cell_rows = 8
    cell_cols = 8
    bin_count = 8
    cut_c = 0.05

    height, width, _ = image.shape

    cut_image = image[int(np.rint(height * cut_c)): int(height - np.rint(height * cut_c)),
                      int(np.rint(width * cut_c)):  int(width - np.rint(width * cut_c))]
    resize_image = resize(cut_image, (64, 64))

    hist = histogram(resize_image, cell_rows, cell_cols, bin_count)

    desc = np.array([], dtype='float64')

    for i in range(0, hist.shape[0] - block_row_cells + 1):
        for j in range(0, hist.shape[1] - block_col_cells + 1):
            block_hist = np.ravel(hist[i:i + block_row_cells, j:j + block_col_cells])
            normed_block_hist = block_hist / sqrt(np.sum(block_hist ** 2) + 1e-10)
            desc = np.append(desc, normed_block_hist)

    return np.ravel(desc)


def fit_and_classify(x_train, y_train, x_test):
    svm_clf = svm.LinearSVC(C=0.5)

    svm_clf.fit(x_train, y_train)
    return svm_clf.predict(x_test)
