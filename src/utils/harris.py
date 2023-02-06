import numpy as np
from numpy.typing import NDArray
from scipy import signal
from utils.filters import blur_filter_img


def harris_corner(img: NDArray[np.float64], window_L: int, k: float, threshold: float) -> tuple[list[list[int]], NDArray[np.float64]]:
    sobelx = np.array([[-1, 0, 1]], dtype=np.float64)
    sobely = sobelx.T

    mat_Ix = signal.convolve2d(img, sobelx, boundary='symm', mode='same')
    mat_Iy = signal.convolve2d(img, sobely, boundary='symm', mode='same')

    gaussian = blur_filter_img(window_L, 0.1)

    mat_A = signal.convolve2d(np.square(mat_Ix), gaussian,
                              boundary='symm', mode='same')
    mat_B = signal.convolve2d(np.square(mat_Iy), gaussian,
                              boundary='symm', mode='same')
    mat_C = signal.convolve2d(mat_Ix * mat_Iy, gaussian,
                              boundary='symm', mode='same')

    det = mat_A * mat_B - np.square(mat_C)
    trace = mat_A + mat_B

    mat_R = det - k * np.square(trace)

    corners = []

    for i in range(1, mat_R.shape[0]-1):
        for j in range(1, mat_R.shape[1]-1):
            if mat_R[i, j] >= threshold:
                if mat_R[i, j] == np.max(mat_R[i-1: i+2, j-1:j+2]):
                    corners.append([i, j])

    return corners, mat_R
