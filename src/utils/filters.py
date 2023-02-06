import numpy as np
from numpy.typing import NDArray


def edge_detection_filter_1d(sigma: float, L: int = 10) -> NDArray[np.float64]:
    seq_h = np.exp(-sigma * (np.arange(1, L+1, dtype=np.float64)))
    weight_C = 1. / np.sum(seq_h)

    return np.concatenate([
        -weight_C * seq_h[::-1],
        np.zeros(1, dtype=np.float64),
        weight_C * seq_h
    ])


def smoother_filter(sigma: float, L: int = 10) -> NDArray[np.float64]:
    seq_h = np.exp(-sigma * np.abs(np.arange(-L, L+1, dtype=np.float64)))
    return seq_h / np.sum(seq_h)


def bilateral_filter(seq_x: NDArray[np.float64], k1: float, k2: float, L: int = 10) -> NDArray[np.float64]:
    len_x = seq_x.shape[0]
    seq_y = np.zeros(len_x, dtype=np.float64)
    for i in range(len_x):
        s, e = max(i - L, 0), min(i + L + 1, len_x)
        weights = (
            np.exp(
                - k1 * np.square(np.arange(s, e, dtype=np.float64) - i)
                - k2 * np.square(seq_x[s: e] - seq_x[i])
            )
        )
        weight_C = 1 / np.sum(weights)
        seq_y[i] = weight_C * np.sum(seq_x[s: e] * weights)
    return seq_y


def match_filter(seq_x: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    len_h = target.shape[0]
    seq_h = np.conj(target)
    mean_h = np.mean(seq_h)
    sig_h = np.sqrt(np.sum(np.square(seq_h - mean_h)))
    if sig_h > 0:
        norm_h = (seq_h - mean_h) / sig_h
    else:
        raise ValueError("the variance of target is equal to 0")

    len_x = seq_x.shape[0]
    x_pad = np.concatenate(
        [seq_x, np.zeros(len_h, dtype=np.float64)],
        dtype=np.float64
    )

    seq_y = np.zeros(len_x, dtype=np.float64)
    for i in range(len_x):
        receptive = x_pad[i: i + len_h]
        mean_x = np.mean(receptive)
        sig_x = np.sqrt(np.sum(np.square(receptive - mean_x)))
        if sig_x > 0:
            seq_y[i] = np.sum((receptive - mean_x) * norm_h) / sig_x
        else:
            seq_y[i] = 0
    return seq_y


def lowpass_mask_img(shape: tuple[int, int], thres_div: float) -> NDArray[np.float64]:
    threshold = min(shape) / thres_div
    mask = np.zeros(shape, dtype=np.float64)
    row_i, col_i = np.mgrid[0: shape[0], 0: shape[1]]
    mask[
        (row_i + col_i <= threshold) |
        (shape[0] - row_i + col_i <= threshold) |
        (row_i + shape[1] - col_i <= threshold) |
        (shape[0] - row_i + shape[1] - col_i <= threshold)
    ] = 1.
    return mask


def blur_filter_img(L: int = 10, sigma: float = 0.1) -> NDArray[np.float64]:
    row_i, col_i = np.mgrid[-L: L+1,
                            -L: L+1].astype(np.float64)
    filt = np.exp(-sigma * (row_i ** 2 + col_i ** 2))
    return filt / np.sum(filt)


def wiener_filter_img(img: NDArray[np.float64], mat_k: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    row, col = img.shape
    row_k, col_k = mat_k.shape

    img_fft = np.fft.fft2(img)
    k_central = np.zeros(img.shape, dtype=np.float64)
    left, top = row // 2 - row_k // 2, col // 2 - col_k // 2
    k_central[left: left + row_k, top: top + col_k] = mat_k
    mat_k_fft = np.fft.fft2(np.fft.ifftshift(k_central))

    mag_k_fft = abs(mat_k_fft)
    phase_k_fft = np.angle(mat_k_fft)

    inv_k_fft = 1 / (
        (C / (mag_k_fft + 1e-5) + mag_k_fft) * np.exp(1j * phase_k_fft)
    )

    return np.fft.ifft2(img_fft * inv_k_fft).real
