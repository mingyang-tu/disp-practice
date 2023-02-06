import numpy as np
from numpy.typing import NDArray


def add_gaussian_noise(x: NDArray[np.float64], sigma: float, amp: float) -> NDArray[np.float64]:
    return x + np.random.normal(0, sigma, x.shape) * amp


def add_gaussian_image(img: NDArray[np.float64], amp: float = 10.) -> NDArray[np.float64]:
    return np.clip(add_gaussian_noise(img, 1, amp), 0, 255)
