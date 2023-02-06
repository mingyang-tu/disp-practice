import numpy as np
from numpy.typing import NDArray


RGB2YCC = np.array([
    [0.299,   0.587,  0.114],
    [-0.169, -0.331,  0.500],
    [0.500,  -0.419, -0.081]
], dtype=np.float64)

YCC2RGB = np.linalg.inv(RGB2YCC)


def cvt_RGB2YCbCr(img: NDArray[np.float64]) -> NDArray[np.float64]:
    img_ycc = img.dot(RGB2YCC.T)
    img_ycc[:, :, [1, 2]] += 127.5
    return img_ycc


def cvt_YCbCr2RGB(img: NDArray[np.float64]) -> NDArray[np.float64]:
    img[:, :, [1, 2]] -= 127.5

    return np.clip(img.dot(YCC2RGB.T), 0, 255)
