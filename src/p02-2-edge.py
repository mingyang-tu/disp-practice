import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils.noise import add_gaussian_noise
from utils.filters import edge_detection_filter_1d


if __name__ == '__main__':
    data_x = np.array(
        [0] * 40 + [1] * 11 + [0] * 29 + [1] * 31 + [0] * 20,
        dtype=np.float64
    )
    data_noisy = add_gaussian_noise(data_x, 1, amp=0.2)
    edf = edge_detection_filter_1d(sigma=0.1, L=10)

    axis_n = np.arange(-30, 101, dtype=np.int64)

    plt.figure("edge")
    plt.subplot(311)
    plt.plot(axis_n, data_x)
    plt.plot(axis_n, data_noisy)
    plt.subplot(312)
    plt.stem(edf)
    plt.subplot(313)
    plt.plot(axis_n, abs(signal.convolve(data_noisy, edf, mode='same')))

    plt.show()
