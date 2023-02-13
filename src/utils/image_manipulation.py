import numpy as np
from numpy.typing import NDArray


def bilinear_interpolation(img: NDArray[np.float64], m1: NDArray[np.float64], n1: NDArray[np.float64]) -> NDArray[np.float64]:
    m0 = np.clip(np.floor(m1), 0, img.shape[0] - 2).astype(int)
    n0 = np.clip(np.floor(n1), 0, img.shape[1] - 2).astype(int)

    a = m1 - m0
    b = n1 - n0

    return (
        (1 - a) * (1 - b) * img[m0, n0] +
        a * (1 - b) * img[m0 + 1, n0] +
        (1 - a) * b * img[m0, n0 + 1] +
        a * b * img[m0 + 1, n0 + 1]
    )


def inv_2x2_mat(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    if mat.shape[0] != 2 or mat.shape[1] != 2:
        raise ValueError("invalid input (mat)")

    (a, b), (c, d) = mat

    det = a * d - b * c
    if det == 0:
        raise ValueError("invalid input (mat)")

    return 1 / det * np.array([[d, -b], [-c, a]], dtype=np.float64)


def resize_image(img: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.float64]:
    row_i, col_i = np.mgrid[0: shape[0], 0: shape[1]].astype(np.float64)

    m1 = row_i * float(img.shape[0]) / float(shape[0])
    n1 = col_i * float(img.shape[1]) / float(shape[1])

    return bilinear_interpolation(img, m1, n1)


def affine_transformation(img: NDArray[np.float64], mat: NDArray[np.float64]) -> NDArray[np.float64]:
    row, col = img.shape
    center = np.array([row // 2, col // 2], dtype=np.float64).reshape(-1, 1, 1)

    row_i, col_i = np.mgrid[0: row, 0: col].astype(np.float64) - center

    (ai, bi), (ci, di) = inv_2x2_mat(mat)
    m1 = ai * row_i + bi * col_i + center[0]
    n1 = ci * row_i + di * col_i + center[1]

    m1_floor = np.floor(m1).astype(int)
    n1_floor = np.floor(n1).astype(int)

    pad_x = max(-np.min(m1_floor), 0), max(np.max(m1_floor) - row + 1, 0)
    pad_y = max(-np.min(n1_floor), 0), max(np.max(n1_floor) - col + 1, 0)
    img_pad = np.pad(img, (pad_x, pad_y))

    return bilinear_interpolation(img_pad, m1 + pad_x[0], n1 + pad_y[0])
