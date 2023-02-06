import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils.noise import add_gaussian_noise
from utils.filters import smoother_filter


if __name__ == '__main__':
    data_x = add_gaussian_noise(
        0.1 * np.arange(-50, 101, dtype=np.float64),
        sigma=1,
        amp=1
    )

    sf = smoother_filter(sigma=0.1, L=5)

    axis_n = np.arange(-50, 101, dtype=np.int64)

    plt.figure("smooth")
    plt.subplot(311)
    plt.plot(axis_n, data_x)
    plt.subplot(312)
    plt.stem(sf)
    plt.subplot(313)
    plt.plot(axis_n, signal.convolve(data_x, sf, mode='same'))

    plt.show()
