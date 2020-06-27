import numpy as np
from scipy.signal import convolve2d


def derivative_e(image):
    e = np.average(image, axis=2, weights=[0.299, 0.587, 0.114])

    x_kernel = np.array([[0., 0., 0.],
                         [1., 0., -1.],
                         [0., 0., 0.]])
    y_kernel = np.array([[0., 1., 0.],
                         [0., 0., 0.],
                         [0., -1., 0.]])

    ix = convolve2d(e, x_kernel, boundary='symm', mode='same')
    iy = convolve2d(e, y_kernel, boundary='symm', mode='same')
    return np.sqrt(ix ** 2 + iy ** 2)


def find_seam(image, mask):
    seams = np.zeros((image.shape[0], image.shape[1]))

    e = derivative_e(image)
    e += e.shape[0] * e.shape[1] * 256.0 * mask

    seams[0] = e[0]

    for height in range(1, e.shape[0]):
        for width in range(0, e.shape[1]):
            upper_neighbors = seams[height-1, max(0, width-1):min(width+2, e.shape[1])]
            seams[height, width] = e[height, width] + np.min(upper_neighbors)

    seam_mask = np.zeros_like(seams)
    height = seams.shape[0] - 1
    seam_ind = np.argmin(seams[height])
    seam_mask[height, seam_ind] = 1

    while height > 0:
        height -= 1
        delta = np.argmin(seams[height, max(0, seam_ind-1):min(seam_ind+2, e.shape[1])]) - 1
        if seam_ind == 0:
            delta += 1
        seam_ind += delta
        seam_mask[height, seam_ind] = 1

    return seam_mask


def horizontal_shrink(image, mask):
    seam_mask = find_seam(image, mask)

    image_shrink = np.zeros((image.shape[0], image.shape[1] - 1, 3))
    mask_shrink = np.zeros((image.shape[0], image.shape[1] - 1))
    for height in range(0, image.shape[0]):
        width = np.argmax(seam_mask[height])

        image_shrink[height, :width] = image[height, :width]
        image_shrink[height, width:] = image[height, width + 1:]

        mask_shrink[height, :width] = mask[height, :width]
        mask_shrink[height, width:] = mask[height, width + 1:]

    return [image_shrink, mask_shrink, seam_mask]


def vertical_shrink(image, mask):
    t_image = np.transpose(image, axes=(1, 0, 2))
    t_mask = np.transpose(mask)

    image_shrink, mask_shrink, seam_mask = horizontal_shrink(t_image, t_mask)
    return [np.transpose(image_shrink, axes=(1, 0, 2)), np.transpose(mask_shrink), np.transpose(seam_mask)]


def horizontal_expand(image, mask):
    seam_mask = find_seam(image, mask)
    mask_expand = np.zeros((image.shape[0], image.shape[1] + 1))
    image_expand = np.zeros((image.shape[0], image.shape[1] + 1, 3))

    for height in range(0, image.shape[0]):
        width = np.argmax(seam_mask[height])

        next_pix = image[height, min(width + 1, image.shape[1] - 1)]
        if width + 1 == image.shape[1]:
            next_pix = image[height, width]

        image_expand[height, :width + 1] = image[height, :width + 1]
        image_expand[height, width + 1] = image[height, width] // 2 + next_pix // 2
        image_expand[height, width + 2:] = image[height, width + 1:]

        mask_expand[height, :width + 1] = mask[height, :width + 1]
        mask_expand[height, width + 1] = mask[height, width]
        mask_expand[height, width + 2:] = mask[height, width + 1:]

    return [image_expand, mask_expand, seam_mask]


def vertical_expand(image, mask):
    t_image = np.transpose(image, axes=(1, 0, 2))
    t_mask = np.transpose(mask)

    image_shrink, mask_shrink, seam_mask = horizontal_expand(t_image, t_mask)
    return [np.transpose(image_shrink, axes=(1, 0, 2)), np.transpose(mask_shrink), np.transpose(seam_mask)]


func_dict = {'horizontal shrink': horizontal_shrink,
             'vertical shrink': vertical_shrink,
             'horizontal expand': horizontal_expand,
             'vertical expand': vertical_expand}


def seam_carve(image, func, mask=None):
    if mask is None:
        mask = np.zeros((image.shape[0], image.shape[1]))

    return func_dict[func](image, mask)
