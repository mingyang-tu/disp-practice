import numpy as np
from numpy.typing import NDArray


def nrmse(img: NDArray[np.float64], noisy: NDArray[np.float64]) -> np.float64:
    return np.sqrt(np.sum(np.square(noisy - img)) / np.sum(np.square(img)))


def snr(img: NDArray[np.float64], noisy: NDArray[np.float64]) -> np.float64:
    return np.mean(np.square(img)) / np.mean(np.square(noisy - img))


def psnr(img: NDArray[np.float64], noisy: NDArray[np.float64]) -> np.float64:
    return 10. * np.log10(255. ** 2 / np.mean(np.square(noisy - img)))


def ssim(img1: NDArray[np.float64], img2: NDArray[np.float64]) -> np.float64:
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    img1_0mean = img1 - mean1
    img2_0mean = img2 - mean2

    var1 = np.mean(np.square(img1_0mean))
    var2 = np.mean(np.square(img2_0mean))
    cov12 = np.mean(img1_0mean * img2_0mean)

    L = 255.
    c1 = c2 = 1 / (L ** (1/2))

    return (
        ((2 * mean1 * mean2 + (c1 * L) ** 2) * (2 * cov12 + (c2 * L) ** 2)) /
        ((mean1 ** 2 + mean2 ** 2 + (c1 * L) ** 2) * (var1 + var2 + (c2 * L) ** 2))
    )


def norm(mat_x: NDArray[np.float64], alpha: float) -> np.float64:
    if alpha == 0:
        return np.sum((mat_x != 0).astype(np.float64))
    elif alpha == np.inf:
        return np.max(np.abs(mat_x))
    else:
        return np.sum(np.abs(mat_x) ** alpha) ** (1 / alpha)


def central_moment_2d(img: NDArray[np.float64], a: int, b: int) -> np.float64:
    row, col = img.shape
    row_i, col_i = np.mgrid[0: row, 0: col].astype(np.float64)

    sum_img = np.sum(img)
    row_avg = np.sum(row_i * img) / sum_img
    col_avg = np.sum(col_i * img) / sum_img

    return np.sum(((row_i - row_avg) ** a) * ((col_i - col_avg) ** b) * img) / sum_img
