import numpy as np
from numpy.typing import NDArray
from scipy import signal


def difference(img: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    img_new = np.zeros(img.shape, dtype=np.float64)
    if axis == 0:
        img_new[1:, :] = img[1:, :] - img[:-1, :]
    elif axis == 1:
        img_new[:, 1:] = img[:, 1:] - img[:, :-1]
    else:
        raise ValueError("invalid input (axis)")
    return img_new


def sobel(img: NDArray[np.float64], angle: int) -> NDArray[np.float64]:
    if angle == 0:
        filt = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    elif angle == 90:
        filt = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    elif angle == 45:
        filt = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]
    elif angle == 135:
        filt = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
    else:
        raise ValueError("invalid input (angle)")

    filt = np.array(filt, dtype=np.float64) / 4

    return signal.convolve2d(img, filt, boundary='symm', mode='same')


def laplacian(img: NDArray[np.float64]) -> NDArray[np.float64]:
    filt = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

    filt = np.array(filt, dtype=np.float64) / 8

    return signal.convolve2d(img, filt, boundary='symm', mode='same')
