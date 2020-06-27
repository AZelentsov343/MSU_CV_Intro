import numpy as np


def mse(I1, I2):
    return np.mean(np.sum((I1 - I2) ** 2))


def cross_correlation(I1, I2):
    denominator = np.sqrt(np.sum(I1 ** 2) * np.sum(I2 ** 2))
    return np.sum(I1*I2)/denominator


def is_better(metric, best_metric, func):
    if func is mse:
        return metric < best_metric
    elif func is cross_correlation:
        return metric > best_metric


def calculate_shifting_metric(image1, image2, shift_h, shift_w, metric_func):

    shifted1 = image1[max(-shift_h, 0):image1.shape[0] - max(shift_h, 0),
                      max(-shift_w, 0):image1.shape[1] - max(shift_w, 0)]
    shifted2 = image2[max(shift_h, 0):image1.shape[0] - max(-shift_h, 0),
                      max(shift_w, 0):image1.shape[1] - max(-shift_w, 0)]

    return metric_func(shifted1, shifted2)


def find_best_shifting(image1, image2, range_shift_h, range_shift_w):
    metric_func = mse
    best_metric = None
    best_h = None
    best_w = None

    for shift_h in range(range_shift_h[0], range_shift_h[1] + 1):
        for shift_w in range(range_shift_w[0], range_shift_w[1] + 1):
            metric = calculate_shifting_metric(image1, image2, shift_h, shift_w, metric_func)
            if best_metric is None or is_better(metric, best_metric, metric_func):
                best_h = shift_h
                best_w = shift_w
                best_metric = metric

    return best_h, best_w


def pyramid_rec(image1, image2):
    if image1.shape[0] < 500 and image2.shape[1] < 500:
        return find_best_shifting(image1, image2, (-15, 15), (-15, 15))
    else:
        small_image1 = image1[::2, ::2]
        small_image2 = image2[::2, ::2]
        best_h, best_w = pyramid_rec(small_image1, small_image2)

        return find_best_shifting(image1, image2, [2*best_h - 1, 2*best_h + 1], [2*best_w - 1, 2*best_w + 1])


def align(image, align_green):
    height = image.shape[0] // 3
    width = image.shape[1]
    blue = image[:height]
    green = image[height:2 * height]
    red = image[2 * height: 3 * height]

    cur_vert = round(0.05 * height)
    cur_horiz = round(0.05 * width)

    blue = blue[cur_vert: height - cur_vert, cur_horiz: width - cur_horiz]
    green = green[cur_vert: height - cur_vert, cur_horiz: width - cur_horiz]
    red = red[cur_vert: height - cur_vert, cur_horiz: width - cur_horiz]

    green_h, green_w = align_green

    shift_h, shift_w = pyramid_rec(blue, green)
    blue = np.roll(blue, (shift_h, shift_w), (0, 1))
    align_blue_h = green_h - shift_h - height
    align_blue_w = green_w - shift_w

    shift_h, shift_w = pyramid_rec(red, green)
    red = np.roll(red, (shift_h, shift_w), (0, 1))
    align_red_h = green_h - shift_h + height
    align_red_w = green_w - shift_w

    return np.stack((red, green, blue), axis=-1).astype(np.uint8), (align_blue_h, align_blue_w), (align_red_h, align_red_w)
